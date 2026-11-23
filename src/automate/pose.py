from __future__ import annotations

import math
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.TopAbs import TopAbs_IN
from OCP.gp import gp_Pnt

from .models import (
    FaceFeature,
    FitType,
    KinematicPairType,
    MatchCandidate,
    MatingConfig,
    PartData,
    SurfaceType,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _rodrigues_align(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return 3x3 SO(3) rotation mapping unit vector u to unit vector v."""
    u = _normalize(u)
    v = _normalize(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1e-9:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(u, ortho))
        return -np.eye(3, dtype=float) + 2.0 * np.outer(axis, axis)
    c = np.cross(u, v)
    s = float(np.linalg.norm(c))
    K = np.array(
        [
            [0.0, -c[2], c[1]],
            [c[2], 0.0, -c[0]],
            [-c[1], c[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + K + (K @ K) * ((1.0 - dot) / (s * s))


def _axis_angle_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    a = _normalize(axis)
    c = math.cos(theta)
    s = math.sin(theta)
    K = np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + s * K + (1.0 - c) * (K @ K)


def _solve_rotation_kabsch(src_vecs: np.ndarray, dst_vecs: np.ndarray) -> np.ndarray:
    """Solve R in SO(3) minimizing sum ||R @ src_i - dst_i||^2."""
    H = src_vecs.T @ dst_vecs
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def _solve_translation_and_free_dofs(
    R: np.ndarray,
    pairs: list[tuple[FaceFeature, FaceFeature]],
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Solve linear least-squares translation t and identify unconstrained translational subspace.
    Returns (t, dof_translation, nullspace_basis_cols).
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for q, t in pairs:
        if q.surface_type == SurfaceType.PLANE and t.surface_type == SurfaceType.PLANE:
            n_t = t.normal
            p_q = R @ q.origin
            rows.append(n_t)
            rhs.append(float(np.dot(t.origin - p_q, n_t)))
        elif q.surface_type == SurfaceType.CYLINDER and t.surface_type == SurfaceType.CYLINDER:
            w_cyl = 50.0
            a_t = t.axis
            P = np.eye(3) - np.outer(a_t, a_t)
            diff = P @ (t.origin - R @ q.origin)
            for row_idx in range(3):
                rows.append(w_cyl * P[row_idx])
                rhs.append(w_cyl * float(diff[row_idx]))
        elif q.surface_type in (SurfaceType.CONE, SurfaceType.SPHERE, SurfaceType.TORUS):
            w_curv = 50.0
            p_q = R @ q.origin
            diff = t.origin - p_q
            for row_idx in range(3):
                e = np.zeros(3)
                e[row_idx] = w_curv
                rows.append(e)
                rhs.append(w_curv * float(diff[row_idx]))
        else:
            # Freeform: sample multiple tangent plane constraints across the surface
            step = max(1, len(t.sample_points) // 16)
            q_sub = q.sample_points[::step]
            t_sub = t.sample_points[::step]
            tn_sub = t.sample_normals[::step]
            for pq, pt, nt in zip(q_sub, t_sub, tn_sub):
                pq_rot = R @ pq
                rows.append(nt)
                rhs.append(float(np.dot(pt - pq_rot, nt)))

    if not rows:
        return np.zeros(3), 3, np.eye(3)

    A = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)

    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    tol = 1e-4 * (S[0] if len(S) > 0 and S[0] > 0 else 1.0)
    rank = int(np.sum(S > tol))
    dof_t = 3 - rank

    # Minimum-norm solution for constrained subspace
    S_inv = np.zeros_like(S)
    for i in range(rank):
        S_inv[i] = 1.0 / S[i]
    t_base = Vt[:rank].T @ (S_inv[:rank] * (U[:, :rank].T @ b))

    null_basis = Vt[rank:].T  # (3, dof_t)
    if dof_t > 0:
        # Center the query group onto the target group along the free sliding DOFs
        q_center = np.mean([R @ q.centroid for q, _ in pairs], axis=0) + t_base
        t_center = np.mean([t.centroid for _, t in pairs], axis=0)
        delta = t_center - q_center
        t_free = null_basis @ (null_basis.T @ delta)
        t_vec = t_base + t_free
    else:
        t_vec = t_base

    return t_vec, dof_t, null_basis


def _compute_rotation_candidates(
    pairs: list[tuple[FaceFeature, FaceFeature]],
) -> list[tuple[np.ndarray, int]]:
    """
    Generate candidate SO(3) rotations R and rotational DOF count.
    Handles fully constrained (>=2 non-collinear directions), 1-DOF axial/planar spin,
    and 3-DOF spherical rotations.
    """
    src_dirs: list[np.ndarray] = []
    dst_dirs: list[np.ndarray] = []
    cyl_pairs: list[tuple[FaceFeature, FaceFeature]] = []

    for q, t in pairs:
        if q.surface_type == SurfaceType.PLANE and t.surface_type == SurfaceType.PLANE:
            src_dirs.append(q.normal)
            dst_dirs.append(-t.normal)
        elif q.surface_type == SurfaceType.CONE and t.surface_type == SurfaceType.CONE:
            src_dirs.append(q.axis)
            dst_dirs.append(t.axis)
        elif q.surface_type == SurfaceType.FREEFORM and q.normal is not None and t.normal is not None:
            src_dirs.append(q.normal)
            dst_dirs.append(-t.normal)
        elif q.surface_type == SurfaceType.CYLINDER and t.surface_type == SurfaceType.CYLINDER:
            cyl_pairs.append((q, t))

    # For cylinders without enough oriented vectors, try both +axis and -axis signs
    if not cyl_pairs:
        if not src_dirs:
            # Pure spheres -> 3-DOF rotation; align centroids if multiple, else identity
            if len(pairs) >= 2:
                q_pts = np.array([q.origin for q, _ in pairs])
                t_pts = np.array([t.origin for _, t in pairs])
                q_c = q_pts - np.mean(q_pts, axis=0)
                t_c = t_pts - np.mean(t_pts, axis=0)
                R0 = _solve_rotation_kabsch(q_c, t_c)
                return [(R0, 1)]
            return [(np.eye(3, dtype=float), 3)]

        S_arr = np.array(src_dirs, dtype=float)
        D_arr = np.array(dst_dirs, dtype=float)
        _, sv, _ = np.linalg.svd(S_arr)
        rank = int(np.sum(sv > 1e-3))
        if rank >= 2:
            return [(_solve_rotation_kabsch(S_arr, D_arr), 0)]

        # 1-DOF spin around single direction D_arr[0], or freeform PCA+ICP alignment
        has_freeform = any(q.surface_type == SurfaceType.FREEFORM for q, _ in pairs)
        if has_freeform:
            return [(R_ff, 0) for R_ff in _freeform_rotation_candidates(pairs)]
        R_align = _rodrigues_align(S_arr[0], D_arr[0])
        rot_axis = _normalize(D_arr[0])
        return [(R_opt, 1) for R_opt in _optimize_1dof_spin(R_align, rot_axis, pairs)]

    # Cylinders present: determine sign options for cylinders
    candidates: list[tuple[np.ndarray, int]] = []
    # Limit sign branching to first cylinder if all cylinders are parallel
    q0, t0 = cyl_pairs[0]
    signs_to_try = [1.0, -1.0]
    for sgn0 in signs_to_try:
        s_list = list(src_dirs)
        d_list = list(dst_dirs)
        s_list.append(q0.axis)
        d_list.append(sgn0 * t0.axis)
        for qc, tc in cyl_pairs[1:]:
            # Choose sign consistent with sgn0 dot relationship if possible
            dot_q = float(np.dot(q0.axis, qc.axis))
            dot_t = float(np.dot(sgn0 * t0.axis, tc.axis))
            sgn = 1.0 if dot_q * dot_t >= 0.0 else -1.0
            s_list.append(qc.axis)
            d_list.append(sgn * tc.axis)

        S_arr = np.array(s_list, dtype=float)
        D_arr = np.array(d_list, dtype=float)
        _, sv, _ = np.linalg.svd(S_arr)
        rank = int(np.sum(sv > 1e-3))
        if rank >= 2:
            candidates.append((_solve_rotation_kabsch(S_arr, D_arr), 0))
        else:
            R_align = _rodrigues_align(S_arr[0], D_arr[0])
            rot_axis = _normalize(D_arr[0])
            for R_opt in _optimize_1dof_spin(R_align, rot_axis, pairs):
                candidates.append((R_opt, 1))
    return candidates


def _freeform_rotation_candidates(
    pairs: list[tuple[FaceFeature, FaceFeature]],
) -> list[np.ndarray]:
    """
    Generate SO(3) rotation candidates for freeform faces using PCA principal frames
    (testing all 4 proper right-handed sign symmetries) plus ICP refinement.
    """
    q0, t0 = next((q, t) for q, t in pairs if q.surface_type == SurfaceType.FREEFORM)
    candidates: list[np.ndarray] = []

    if q0.pca_axes is not None and t0.pca_axes is not None:
        # 4 proper SO(3) sign flips of PCA axes: (+,+,+), (+,-,-), (-,+,-), (-,-,+)
        for signs in ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)):
            S = np.diag(signs) @ t0.pca_axes
            R_init = S.T @ q0.pca_axes
            if np.linalg.det(R_init) < 0:
                S[2, :] *= -1.0
                R_init = S.T @ q0.pca_axes
            candidates.append(R_init)

    if q0.normal is not None and t0.normal is not None:
        R_n = _rodrigues_align(q0.normal, -t0.normal)
        candidates.extend(_optimize_1dof_spin(R_n, _normalize(-t0.normal), pairs))

    # Refine each candidate with 4 iterations of point+normal Kabsch ICP
    t_tree = cKDTree(t0.sample_points)
    q_c = q0.sample_points - q0.centroid
    best_R = candidates[0] if candidates else np.eye(3)
    best_err = 1e18

    for R_cand in candidates:
        R_cur = R_cand.copy()
        for _ in range(4):
            t_vec = t0.centroid - R_cur @ q0.centroid
            q_w = (q0.sample_points @ R_cur.T) + t_vec
            _, idxs = t_tree.query(q_w)
            t_matched = t0.sample_points[idxs] - t0.centroid
            tn_matched = -t0.sample_normals[idxs]
            # Combine centered points and scaled normals for joint orientation refinement
            scale = max(1.0, float(np.std(q_c)))
            src_aug = np.vstack([q_c, scale * q0.sample_normals])
            dst_aug = np.vstack([t_matched, scale * tn_matched])
            R_cur = _solve_rotation_kabsch(src_aug, dst_aug)

        t_vec = t0.centroid - R_cur @ q0.centroid
        q_w = (q0.sample_points @ R_cur.T) + t_vec
        dists, idxs = t_tree.query(q_w)
        n_dots = np.sum((q0.sample_normals @ R_cur.T) * (-t0.sample_normals[idxs]), axis=1)
        err = float(np.mean(dists * dists)) + 5.0 * float(np.mean(1.0 - n_dots))
        if err < best_err:
            best_err = err
            best_R = R_cur

    return [best_R]


def _optimize_1dof_spin(
    R_base: np.ndarray,
    rot_axis: np.ndarray,
    pairs: list[tuple[FaceFeature, FaceFeature]],
) -> list[np.ndarray]:
    """
    When rotation has 1 free DOF around rot_axis (e.g. single plane, single cylinder,
    parallel opposing planes, or belt arc on pulley), sample angles and pick best overlap/centroid alignment.
    """
    best_R = R_base
    best_score = -1e18

    # Coarse 24-angle scan (every 15 deg) + local refinement
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    # Also include PCA-aligned angle for single planar faces if available
    if len(pairs) == 1 and pairs[0][0].pca_axes is not None and pairs[0][1].pca_axes is not None:
        q0, t0 = pairs[0]
        q_maj = R_base @ q0.pca_axes[0]
        t_maj = t0.pca_axes[0]
        # Project onto plane orthogonal to rot_axis
        q_proj = q_maj - np.dot(q_maj, rot_axis) * rot_axis
        t_proj = t_maj - np.dot(t_maj, rot_axis) * rot_axis
        if np.linalg.norm(q_proj) > 1e-6 and np.linalg.norm(t_proj) > 1e-6:
            q_proj = _normalize(q_proj)
            t_proj = _normalize(t_proj)
            ang = math.atan2(float(np.dot(rot_axis, np.cross(q_proj, t_proj))), float(np.dot(q_proj, t_proj)))
            angles = np.concatenate([angles, [ang, ang + math.pi]])

    target_trees = [cKDTree(t.sample_points) for _, t in pairs]
    for theta in angles:
        R_try = _axis_angle_matrix(rot_axis, float(theta)) @ R_base
        t_try, _, _ = _solve_translation_and_free_dofs(R_try, pairs)
        # Score by negative mean squared distance from transformed query points to target points
        score = 0.0
        for (q, _), tree in zip(pairs, target_trees):
            q_pts = (q.sample_points @ R_try.T) + t_try
            dists, _ = tree.query(q_pts)
            score -= float(np.mean(dists * dists))
        if score > best_score:
            best_score = score
            best_R = R_try

    return [best_R]


def _signed_point_to_face_surface_gap(
    pts_world: np.ndarray,
    target_face: FaceFeature,
) -> np.ndarray:
    """
    Compute signed distance from world points to target_face's underlying surface,
    measured along target_face's outward normal (>0 = in empty space / clearance,
    0 = coincident, <0 = solid penetration / interference).
    """
    st = target_face.surface_type
    if st == SurfaceType.PLANE:
        return (pts_world - target_face.origin) @ target_face.normal

    if st == SurfaceType.CYLINDER:
        rel = pts_world - target_face.origin
        ax = target_face.axis
        radial = rel - np.outer(rel @ ax, ax)
        r_pts = np.linalg.norm(radial, axis=1)
        # sense == +1 (convex shaft): outward normal points +r -> gap = r_pts - R
        # sense == -1 (concave bore): outward normal points -r -> gap = R - r_pts
        return target_face.sense * (r_pts - target_face.radius)

    if st == SurfaceType.SPHERE:
        r_pts = np.linalg.norm(pts_world - target_face.origin, axis=1)
        return target_face.sense * (r_pts - target_face.radius)

    if st == SurfaceType.CONE:
        rel = pts_world - target_face.origin
        ax = target_face.axis
        h = rel @ ax
        radial = rel - np.outer(h, ax)
        r_pts = np.linalg.norm(radial, axis=1)
        alpha = target_face.half_angle
        # Signed distance to cone surface along outward normal:
        # On cone: r = h * tan(alpha). Outward radial component has sign = target_face.sense.
        # Exact perpendicular distance to cone line in (r, h) plane:
        # signed radial excess = sense * (r_pts - h * tan(alpha)) * cos(alpha)
        return target_face.sense * (r_pts * math.cos(alpha) - h * math.sin(alpha))

    # Freeform / Torus fallback: nearest sampled point projected onto sample normal
    tree = cKDTree(target_face.sample_points)
    _, idxs = tree.query(pts_world)
    nearest_p = target_face.sample_points[idxs]
    nearest_n = target_face.sample_normals[idxs]
    return np.sum((pts_world - nearest_p) * nearest_n, axis=1)


def _analytic_target_normals(pts_world: np.ndarray, t: FaceFeature) -> np.ndarray:
    """Evaluate exact outward normals on target_face's underlying surface at pts_world."""
    st = t.surface_type
    if st == SurfaceType.PLANE:
        return np.tile(t.normal, (len(pts_world), 1))

    if st == SurfaceType.CYLINDER:
        rel = pts_world - t.origin
        ax = t.axis
        radial = rel - np.outer(rel @ ax, ax)
        norms = np.linalg.norm(radial, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return t.sense * (radial / norms)

    if st == SurfaceType.SPHERE:
        rel = pts_world - t.origin
        norms = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        return t.sense * (rel / norms)

    if st == SurfaceType.CONE:
        rel = pts_world - t.origin
        ax = t.axis
        h = rel @ ax
        radial = rel - np.outer(h, ax)
        norms = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        r_hat = radial / norms
        alpha = t.half_angle
        n_raw = r_hat * math.cos(alpha) - np.outer(np.full(len(pts_world), math.sin(alpha)), ax)
        n_len = np.maximum(np.linalg.norm(n_raw, axis=1, keepdims=True), 1e-12)
        return t.sense * (n_raw / n_len)

    tree = cKDTree(t.sample_points)
    _, idxs = tree.query(pts_world)
    return t.sample_normals[idxs]


def _dist_points_to_occt_face(pts: np.ndarray, occt_face, max_samples: int = 25) -> np.ndarray:
    """Compute exact 3D Euclidean distances from subsampled points to a trimmed B-Rep face."""
    if len(pts) == 0 or occt_face is None:
        return np.zeros(0, dtype=float)
    step = max(1, len(pts) // max_samples)
    sub = pts[::step][:max_samples]
    dists = np.empty(len(sub), dtype=float)
    for idx, p in enumerate(sub):
        vtx = BRepBuilderAPI_MakeVertex(gp_Pnt(float(p[0]), float(p[1]), float(p[2]))).Shape()
        ext = BRepExtrema_DistShapeShape(vtx, occt_face)
        dists[idx] = float(ext.Value()) if ext.IsDone() else 1e6
    return dists


def _compute_overlap_coverage(
    q_pts_world: np.ndarray,
    q_face: FaceFeature,
    t_face: FaceFeature,
    R: np.ndarray,
    t_vec: np.ndarray,
    tol_mm: float,
    normal_gap_tol: float,
) -> tuple[float, float]:
    """
    Compute exact trimmed-face overlap fractions (cov_q_in_t, cov_t_in_q)
    using Open CASCADE BRepExtrema_DistShapeShape so coverage is invariant to
    anisotropic aspect ratios or UV grid spacing.
    """
    eff_tol = math.sqrt(tol_mm ** 2 + normal_gap_tol ** 2)

    if t_face.occt_face is not None:
        dq = _dist_points_to_occt_face(q_pts_world, t_face.occt_face)
        cov_q = float(np.mean(dq <= eff_tol)) if len(dq) else 0.0
    else:
        t_tree = cKDTree(t_face.sample_points)
        dq, _ = t_tree.query(q_pts_world)
        cov_q = float(np.mean(dq <= eff_tol))

    if q_face.occt_face is not None:
        # Map target sample points back into query part's frame to query q_face.occt_face
        t_pts_in_q = (t_face.sample_points - t_vec) @ R
        dt = _dist_points_to_occt_face(t_pts_in_q, q_face.occt_face)
        cov_t = float(np.mean(dt <= eff_tol)) if len(dt) else 0.0
    else:
        q_tree = cKDTree(q_pts_world)
        dt, _ = q_tree.query(t_face.sample_points)
        cov_t = float(np.mean(dt <= eff_tol))

    return cov_q, cov_t


def _check_solid_collision(
    R: np.ndarray,
    t_vec: np.ndarray,
    pairs: list[tuple[FaceFeature, FaceFeature]],
    target_parts: dict[int, PartData],
    probe_depth_mm: float,
    query_part: Optional[PartData] = None,
) -> bool:
    """
    Return True if placement causes solid penetration (query material inside target solid).
    Probes:
      1. Overlapping points on selected query faces pushed slightly inward (-depth * n_w).
      2. Interior material points on unselected faces of query_part (rejects upside-down /
         backward insertions where the part body/handle penetrates the target solid).
    """
    unique_target_parts = {t.part_id for _, t in pairs}
    classifiers = {}
    for pid in unique_target_parts:
        part = target_parts.get(pid)
        if part is not None:
            classifiers[pid] = BRepClass3d_SolidClassifier(part.shape)

    if not classifiers:
        return False

    depth = min(probe_depth_mm, 0.12)
    for q, t in pairs:
        clf = classifiers.get(t.part_id)
        if clf is None:
            continue
        step = max(1, len(q.sample_points) // 7)
        q_sub_pts = q.sample_points[::step]
        q_sub_nrms = q.sample_normals[::step]
        q_sub_w = (q_sub_pts @ R.T) + t_vec
        sgaps = _signed_point_to_face_surface_gap(q_sub_w, t)
        for p_w, n, sg in zip(q_sub_w, q_sub_nrms, sgaps):
            # Only probe points whose orthogonal projection lands strictly inside t.occt_face
            if t.occt_face is not None:
                vtx = BRepBuilderAPI_MakeVertex(gp_Pnt(float(p_w[0]), float(p_w[1]), float(p_w[2]))).Shape()
                ext = BRepExtrema_DistShapeShape(vtx, t.occt_face)
                if not ext.IsDone() or float(ext.Value()) > abs(float(sg)) + 0.12:
                    continue
            n_w = R @ n
            probe = p_w - depth * n_w
            clf.Perform(gp_Pnt(float(probe[0]), float(probe[1]), float(probe[2])), 1e-4)
            if clf.State() == TopAbs_IN:
                return True

    # Check unselected query_part body faces against target solid(s)
    if query_part is not None and len(query_part.faces) > len(pairs):
        sel_ids = {q.local_face_id for q, _ in pairs}
        body_depth = 0.20
        for f in query_part.faces:
            if f.local_face_id in sel_ids or len(f.sample_points) == 0:
                continue
            step = max(1, len(f.sample_points) // 4)
            pts_in = f.sample_points[::step] - body_depth * f.sample_normals[::step]
            pts_w = (pts_in @ R.T) + t_vec
            for clf in classifiers.values():
                for pw in pts_w:
                    clf.Perform(gp_Pnt(float(pw[0]), float(pw[1]), float(pw[2])), 1e-4)
                    if clf.State() == TopAbs_IN:
                        return True

    return False


def _determine_kinematic_pair(
    pairs: list[tuple[FaceFeature, FaceFeature]],
    dof_t: int,
    dof_r_base: int,
    R: np.ndarray,
) -> tuple[KinematicPairType, int, int]:
    """Classify the kinematic pair from constraint DOFs."""
    dirs: list[np.ndarray] = []
    for _, t in pairs:
        if t.surface_type == SurfaceType.PLANE and t.normal is not None:
            dirs.append(t.normal)
        elif t.surface_type in (SurfaceType.CYLINDER, SurfaceType.CONE, SurfaceType.TORUS) and t.axis is not None:
            dirs.append(t.axis)
        elif t.surface_type == SurfaceType.FREEFORM and len(t.sample_normals) > 0:
            step = max(1, len(t.sample_normals) // 16)
            dirs.extend(t.sample_normals[::step].tolist())

    if not dirs:
        dof_r = 3
    else:
        _, sv, _ = np.linalg.svd(np.array(dirs, dtype=float))
        rank_r = int(np.sum(sv > 1e-3))
        dof_r = max(0, 3 - (2 if rank_r >= 2 else rank_r))
        # Wait: 1 direction constrains 2 tilt DOFs -> leaves 1 rotation DOF around itself!
        if rank_r >= 2:
            dof_r = 0
        elif rank_r == 1:
            dof_r = 1
        else:
            dof_r = 3

    if dof_t == 0 and dof_r == 0:
        return KinematicPairType.RIGID_0DOF, 0, 0
    if dof_t == 1 and dof_r == 0:
        return KinematicPairType.PRISMATIC_SLIDE, 1, 0
    if dof_t == 0 and dof_r == 1:
        return KinematicPairType.REVOLUTE, 0, 1
    if dof_t == 1 and dof_r == 1:
        return KinematicPairType.CYLINDRICAL, 1, 1
    if dof_t == 2 and dof_r == 1:
        return KinematicPairType.PLANAR, 2, 1
    if dof_t == 0 and dof_r == 3:
        return KinematicPairType.SPHERICAL, 0, 3
    return KinematicPairType.GENERAL_CONSTRAINED, dof_t, dof_r


def _opposing_parallel_gap(
    pairs: list[tuple[FaceFeature, FaceFeature]],
) -> Optional[float]:
    """
    Compute the invariant opposing-plane clearance (d_t + d_q) across any opposing
    parallel plane pairs, or radial clearance (R_concave - R_convex) for cylinders/spheres.
    This metric is independent of centering!
    """
    gaps: list[float] = []
    n_pairs = len(pairs)
    for i in range(n_pairs):
        q1, t1 = pairs[i]
        if q1.surface_type in (SurfaceType.CYLINDER, SurfaceType.SPHERE) and t1.surface_type == q1.surface_type:
            # R_concave - R_convex
            r_diff = (t1.radius - q1.radius) if t1.sense < 0 else (q1.radius - t1.radius)
            gaps.append(float(r_diff))
        for j in range(i + 1, n_pairs):
            q2, t2 = pairs[j]
            if q1.surface_type == SurfaceType.PLANE and q2.surface_type == SurfaceType.PLANE:
                if float(np.dot(q1.normal, q2.normal)) < -0.995:
                    d_q = float(np.dot(q2.origin - q1.origin, q1.normal))
                    d_t = float(np.dot(t2.origin - t1.origin, t1.normal))
                    gaps.append(d_t + d_q)
    if not gaps:
        return None
    return float(max(gaps))


def solve_and_verify_match(
    pairs: list[tuple[FaceFeature, FaceFeature]],
    target_parts: dict[int, PartData],
    cfg: MatingConfig,
    match_id: int = 0,
    query_part: Optional[PartData] = None,
) -> Optional[MatchCandidate]:
    """
    Solve SE(3) alignment for a correspondence set pairs=[(q_i, t_i)],
    verify surface residuals, trimmed overlap, and solid non-penetration,
    and classify Exact/Interference vs Sliding/Clearance fit.
    """
    rot_candidates = _compute_rotation_candidates(pairs)
    if isinstance(rot_candidates, tuple):
        rot_candidates = [(rot_candidates[0], rot_candidates[1])]  # type: ignore[index]

    best_candidate: Optional[MatchCandidate] = None

    for R, dof_r_base in rot_candidates:
        t_vec_base, dof_t, null_basis = _solve_translation_and_free_dofs(R, pairs)

        # If 1-DOF sliding axis exists and query_part has different axial span than target pocket,
        # also test end-flush shifts so a longer key tip seats above a blind pocket floor.
        t_candidates = [t_vec_base]
        if dof_t == 1 and null_basis.shape[1] >= 1:
            u_slide = null_basis[:, 0]
            q_pts_all = np.vstack([(q.sample_points @ R.T) + t_vec_base for q, _ in pairs])
            t_pts_all = np.vstack([t.sample_points for _, t in pairs])
            q_span = 1.25 * float(np.ptp(q_pts_all @ u_slide))
            t_span = 1.25 * float(np.ptp(t_pts_all @ u_slide))
            delta = 0.5 * abs(q_span - t_span) + 0.25
            if delta > 0.35:
                t_candidates.append(t_vec_base + delta * u_slide)
                t_candidates.append(t_vec_base - delta * u_slide)

        for t_vec in t_candidates:
            all_gaps: list[float] = []
            face_mean_gaps: list[float] = []
            all_normal_errs: list[float] = []
            all_covs: list[float] = []
            all_sym_covs: list[float] = []
            valid = True

            # Characteristic max clearance for this feature group
            scales: list[float] = []
            for q, t in pairs:
                if q.radius is not None:
                    scales.append(2.0 * q.radius)
                if q.extent_u is not None:
                    scales.append(q.extent_u)
            char_scale = max(scales) if scales else 10.0
            eff_max_gap = min(cfg.max_clearance_mm, max(0.35, cfg.relative_clearance_ratio * char_scale))

            for q, t in pairs:
                q_pts_w = (q.sample_points @ R.T) + t_vec
                q_nrm_w = q.sample_normals @ R.T

                # Exact analytic normal opposition check
                t_nrms = _analytic_target_normals(q_pts_w, t)
                dots = np.clip(-np.sum(q_nrm_w * t_nrms, axis=1), -1.0, 1.0)
                ang_errs = np.degrees(np.arccos(dots))
                mean_ang = float(np.mean(ang_errs))
                if mean_ang > cfg.angle_tol_deg:
                    valid = False
                    break
                all_normal_errs.append(mean_ang)

                # Signed surface gap check
                sgaps = _signed_point_to_face_surface_gap(q_pts_w, t)
                min_g = float(np.min(sgaps))
                max_g = float(np.max(sgaps))
                mean_g = float(np.mean(sgaps))

                if min_g < -cfg.interference_tol_mm:
                    valid = False
                    break
                if mean_g > eff_max_gap or max_g > eff_max_gap + 0.5:
                    valid = False
                    break

                all_gaps.extend(sgaps.tolist())
                face_mean_gaps.append(mean_g)

                # Exact trimmed B-Rep overlap check
                cov_q, cov_t = _compute_overlap_coverage(
                    q_pts_w,
                    q,
                    t,
                    R,
                    t_vec,
                    tol_mm=cfg.edge_overlap_tol_mm,
                    normal_gap_tol=max(0.5, abs(mean_g) + 0.3),
                )
                cov = max(cov_q, cov_t)
                if cov < cfg.min_coverage:
                    valid = False
                    break
                all_covs.append(cov)
                all_sym_covs.append(0.5 * (cov_q + cov_t))

            if not valid:
                continue

            # Solid collision probe (both selected face overlap and unselected query body)
            if cfg.check_solid_collision:
                if _check_solid_collision(
                    R, t_vec, pairs, target_parts, cfg.collision_probe_depth_mm, query_part=query_part
                ):
                    continue

            min_gap = float(np.min(all_gaps))
            max_gap = float(np.max(all_gaps))
            mean_gap = float(np.mean(all_gaps))
            max_face_gap = float(np.max(face_mean_gaps))
            mean_norm_err = float(np.mean(all_normal_errs))
            mean_cov = float(np.mean(all_covs))
            mean_sym_cov = float(np.mean(all_sym_covs))
            opp_gap = _opposing_parallel_gap(pairs)

            kin_pair, k_dof_t, k_dof_r = _determine_kinematic_pair(pairs, dof_t, dof_r_base, R)

            # Fit classification: Exact/Interference vs Sliding/Clearance
            gap_indicator = opp_gap if opp_gap is not None else mean_gap
            if min_gap < -0.01 or (opp_gap is not None and opp_gap < -0.01):
                fit_type = FitType.INTERFERENCE_FIT
            elif gap_indicator <= cfg.exact_fit_tol_mm and max_face_gap <= cfg.exact_fit_tol_mm + 0.05:
                fit_type = FitType.EXACT_FIT
            else:
                fit_type = FitType.SLIDING_CLEARANCE_FIT

            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = t_vec

            # Score prioritizes high overlap, area symmetry, low normal error, and tight fit
            score = (
                mean_cov * 0.45
                + mean_sym_cov * 0.20
                + max(0.0, 1.0 - mean_norm_err / max(1.0, cfg.angle_tol_deg)) * 0.20
                + max(0.0, 1.0 - abs(mean_gap) / max(0.5, eff_max_gap)) * 0.15
            )

            part_ids = sorted({t.part_id for _, t in pairs})
            part_names = []
            seen_pid = set()
            for _, t in pairs:
                if t.part_id not in seen_pid:
                    seen_pid.add(t.part_id)
                    part_names.append(t.part_name)

            reasoning = (
                f"{fit_type.value} | {kin_pair.value} (T:{k_dof_t}, R:{k_dof_r}) | "
                f"mean_gap={mean_gap:.3f}mm, opp_clear={opp_gap:.3f}mm, "
                if opp_gap is not None
                else f"{fit_type.value} | {kin_pair.value} (T:{k_dof_t}, R:{k_dof_r}) | mean_gap={mean_gap:.3f}mm, "
            )
            reasoning += f"norm_err={mean_norm_err:.2f}deg, overlap={mean_cov:.2f}"

            cand = MatchCandidate(
                match_id=match_id,
                part_ids=part_ids,
                part_names=part_names,
                query_face_ids=[q.local_face_id for q, _ in pairs],
                target_global_face_ids=[t.global_id for _, t in pairs],
                target_local_face_ids=[t.local_face_id for _, t in pairs],
                target_part_ids=[t.part_id for _, t in pairs],
                transform_4x4=T,
                fit_type=fit_type,
                kinematic_pair=kin_pair,
                dof_translation=k_dof_t,
                dof_rotation=k_dof_r,
                mean_gap_mm=mean_gap,
                min_gap_mm=min_gap,
                max_gap_mm=max_gap,
                opposing_clearance_mm=opp_gap,
                normal_error_deg=mean_norm_err,
                overlap_score=mean_cov,
                score=score,
                reasoning=reasoning,
            )

            if best_candidate is None or cand.score > best_candidate.score:
                best_candidate = cand
            break  # Found valid non-colliding shift along 1-DOF axis

    return best_candidate
