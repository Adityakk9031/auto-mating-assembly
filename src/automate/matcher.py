from __future__ import annotations

import math
from typing import Optional
import numpy as np

from .models import (
    AssemblyData,
    FaceFeature,
    MatchCandidate,
    MatingConfig,
    PartData,
    SurfaceType,
)
from .pose import solve_and_verify_match
from .step_io import is_same_part_instance


def _bbox_min_max_dist(
    amin: np.ndarray, amax: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> tuple[float, float]:
    """Return (min_dist, max_dist) between two 3D axis-aligned bounding boxes."""
    dmin_sq = 0.0
    dmax_sq = 0.0
    for k in range(3):
        if amax[k] < bmin[k]:
            d = bmin[k] - amax[k]
            dmin_sq += d * d
        elif bmax[k] < amin[k]:
            d = amin[k] - bmax[k]
            dmin_sq += d * d
        far = max(abs(amax[k] - bmin[k]), abs(bmax[k] - amin[k]))
        dmax_sq += far * far
    return math.sqrt(dmin_sq), math.sqrt(dmax_sq)


def _line_to_line_dist(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
    """Shortest 3D distance between two infinite lines (p1 + s*d1) and (p2 + t*d2)."""
    cross = np.cross(d1, d2)
    denom = float(np.linalg.norm(cross))
    diff = p2 - p1
    if denom < 1e-6:
        # Parallel lines
        perp = diff - float(np.dot(diff, d1)) * d1
        return float(np.linalg.norm(perp))
    return abs(float(np.dot(diff, cross))) / denom


def _eff_max_gap(char_size: float, cfg: MatingConfig) -> float:
    return min(cfg.max_clearance_mm, max(0.35, cfg.relative_clearance_ratio * char_size))


def _single_face_compatible(q: FaceFeature, t: FaceFeature, cfg: MatingConfig) -> bool:
    if q.surface_type != t.surface_type:
        return False

    st = q.surface_type
    if st == SurfaceType.PLANE:
        # Reject extreme area mismatches (> 50x) unless it could be a valid sliding rail
        if q.area > 0 and t.area > 0:
            ratio = min(q.area, t.area) / max(q.area, t.area)
            if ratio < 0.01:
                return False
        return True

    if st in (SurfaceType.CYLINDER, SurfaceType.SPHERE):
        if q.sense * t.sense != -1:
            return False
        r_conv = q.radius if q.sense > 0 else t.radius
        r_conc = t.radius if t.sense < 0 else q.radius
        if r_conv is None or r_conc is None:
            return False
        clearance = r_conc - r_conv
        eff_max = _eff_max_gap(2.0 * r_conv, cfg)
        return -cfg.interference_tol_mm <= clearance <= eff_max

    if st == SurfaceType.CONE:
        if q.sense * t.sense != -1:
            return False
        if q.half_angle is None or t.half_angle is None:
            return False
        if abs(q.half_angle - t.half_angle) > math.radians(cfg.cone_angle_tol_deg):
            return False
        if q.r_min is not None and t.r_max is not None:
            eff_max = _eff_max_gap(2.0 * (q.radius or 10.0), cfg)
            # Radial intervals should overlap within clearance
            if q.r_max + eff_max < t.r_min or t.r_max + eff_max < q.r_min:
                return False
        return True

    if st == SurfaceType.TORUS:
        if q.sense * t.sense != -1:
            return False
        if q.major_radius is None or t.major_radius is None:
            return False
        eff_max = _eff_max_gap(2.0 * q.major_radius, cfg)
        return (
            abs(q.major_radius - t.major_radius) <= eff_max
            and abs((q.minor_radius or 0.0) - (t.minor_radius or 0.0)) <= eff_max
        )

    # FREEFORM
    if q.sense != 0 and t.sense != 0 and q.sense * t.sense != -1:
        return False
    if q.area > 0 and t.area > 0:
        ratio = min(q.area, t.area) / max(q.area, t.area)
        if ratio < 0.05:
            return False
    return True


def _pairwise_compatible(
    q1: FaceFeature,
    q2: FaceFeature,
    t1: FaceFeature,
    t2: FaceFeature,
    cfg: MatingConfig,
) -> bool:
    if t1.global_id == t2.global_id:
        return False

    # 1. Bounding-box spatial proximity consistency
    q_min_d, q_max_d = _bbox_min_max_dist(q1.bbox_min, q1.bbox_max, q2.bbox_min, q2.bbox_max)
    t_min_d, t_max_d = _bbox_min_max_dist(t1.bbox_min, t1.bbox_max, t2.bbox_min, t2.bbox_max)
    pad = cfg.max_clearance_mm + 1.5
    if t_min_d > q_max_d + pad or q_min_d > t_max_d + pad:
        return False

    # 2. SE(3)-invariant directional angle check
    ang_tol_rad = math.radians(cfg.angle_tol_deg)

    def _vec_and_mate_sign(f: FaceFeature) -> tuple[Optional[np.ndarray], int]:
        if f.surface_type in (SurfaceType.PLANE, SurfaceType.FREEFORM):
            return f.normal, -1
        if f.surface_type == SurfaceType.CONE:
            return f.axis, +1
        if f.surface_type in (SurfaceType.CYLINDER, SurfaceType.TORUS):
            return f.axis, 0
        return None, 0

    vq1, sq1 = _vec_and_mate_sign(q1)
    vq2, sq2 = _vec_and_mate_sign(q2)
    vt1, _ = _vec_and_mate_sign(t1)
    vt2, _ = _vec_and_mate_sign(t2)
    if vq1 is not None and vq2 is not None and vt1 is not None and vt2 is not None:
        dq = float(np.clip(np.dot(vq1, vq2), -1.0, 1.0))
        dt = float(np.clip(np.dot(vt1, vt2), -1.0, 1.0))
        if sq1 != 0 and sq2 != 0:
            if abs(math.acos(dq) - math.acos(float(np.clip((sq1 * sq2) * dt, -1.0, 1.0)))) > ang_tol_rad:
                return False
        else:
            if abs(math.acos(abs(dq)) - math.acos(abs(dt))) > ang_tol_rad:
                return False

    # 3. Pairwise metric / signed offset constraints
    st1, st2 = q1.surface_type, q2.surface_type

    if st1 == SurfaceType.PLANE and st2 == SurfaceType.PLANE:
        dot_n = float(np.dot(q1.normal, q2.normal))
        par_thresh = math.cos(math.radians(cfg.parallel_tol_deg))
        if abs(dot_n) >= par_thresh:
            d_q = float(np.dot(q2.origin - q1.origin, q1.normal))
            d_t = float(np.dot(t2.origin - t1.origin, t1.normal))
            g = d_t + d_q
            if dot_n < 0.0:
                # Parallel opposing planes (key width vs slot width, or vice tongue vs groove)
                char_w = max(abs(d_q), abs(d_t), 1.0)
                eff_max = _eff_max_gap(char_w, cfg)
                if not (-cfg.interference_tol_mm <= g <= eff_max):
                    return False
            else:
                # Parallel same-direction planes (step height)
                if abs(g) > cfg.max_clearance_mm:
                    return False
        else:
            # Non-parallel planes: signed half-space overlap kills convex-to-convex false positives
            sq_12 = (q2.sample_points - q1.origin) @ q1.normal
            st_12 = (t2.sample_points - t1.origin) @ t1.normal
            sq_21 = (q1.sample_points - q2.origin) @ q2.normal
            st_21 = (t1.sample_points - t2.origin) @ t2.normal

            tol_hs = cfg.max_clearance_mm + 0.5
            # Interval [-max(st), -min(st)] must overlap [min(sq), max(sq)]
            ov1 = min(float(np.max(sq_12)), -float(np.min(st_12))) - max(float(np.min(sq_12)), -float(np.max(st_12)))
            ov2 = min(float(np.max(sq_21)), -float(np.min(st_21))) - max(float(np.min(sq_21)), -float(np.max(st_21)))
            if ov1 < -tol_hs or ov2 < -tol_hs:
                return False

    elif st1 == SurfaceType.CYLINDER and st2 == SurfaceType.CYLINDER:
        d_q = _line_to_line_dist(q1.origin, q1.axis, q2.origin, q2.axis)
        d_t = _line_to_line_dist(t1.origin, t1.axis, t2.origin, t2.axis)
        if abs(d_q - d_t) > cfg.axis_dist_tol_mm:
            return False

    elif {st1, st2} == {SurfaceType.PLANE, SurfaceType.CYLINDER}:
        qp, qc = (q1, q2) if st1 == SurfaceType.PLANE else (q2, q1)
        tp, tc = (t1, t2) if st1 == SurfaceType.PLANE else (t2, t1)
        dot_na = abs(float(np.dot(qp.normal, qc.axis)))
        if dot_na < 0.25:
            d_q = float(np.dot(qc.origin - qp.origin, qp.normal))
            d_t = float(np.dot(tc.origin - tp.origin, tp.normal))
            eff_max = _eff_max_gap(2.0 * (qc.radius or 10.0), cfg)
            if abs(d_t + d_q) > eff_max:
                return False
        elif dot_na > 0.95:
            # Plane perpendicular to cylinder axis (shoulder / end stop):
            # Under R n_q = -n_t, the axial half-space of the cylinder centroid
            # relative to the plane must have opposite signs on query vs target
            sq = float(np.dot(qc.centroid - qp.origin, qp.normal))
            st = float(np.dot(tc.centroid - tp.origin, tp.normal))
            if abs(sq) > 0.5 and abs(st) > 0.5 and (sq * st > 0.0):
                return False

    elif st1 == SurfaceType.CONE and st2 == SurfaceType.CONE:
        d_q = _line_to_line_dist(q1.origin, q1.axis, q2.origin, q2.axis)
        d_t = _line_to_line_dist(t1.origin, t1.axis, t2.origin, t2.axis)
        if abs(d_q - d_t) > cfg.axis_dist_tol_mm:
            return False
        if d_q < 0.5 and abs(float(np.dot(q1.axis, q2.axis))) > 0.99:
            sep_q = float(np.dot(q2.origin - q1.origin, q1.axis))
            sep_t = float(np.dot(t2.origin - t1.origin, t1.axis))
            alpha = max(0.1, min(q1.half_angle or 0.3, q2.half_angle or 0.3))
            axial_tol = cfg.max_clearance_mm / math.sin(alpha) + 0.5
            if abs(sep_q - sep_t) > axial_tol:
                return False

    elif SurfaceType.SPHERE in (st1, st2):
        if st1 == SurfaceType.SPHERE and st2 == SurfaceType.SPHERE:
            dist_q = float(np.linalg.norm(q2.origin - q1.origin))
            dist_t = float(np.linalg.norm(t2.origin - t1.origin))
            if abs(dist_q - dist_t) > cfg.max_clearance_mm + 0.5:
                return False
        else:
            qs, qo = (q1, q2) if st1 == SurfaceType.SPHERE else (q2, q1)
            ts, to = (t1, t2) if st1 == SurfaceType.SPHERE else (t2, t1)
            if qo.surface_type == SurfaceType.PLANE:
                dq = float(np.dot(qs.origin - qo.origin, qo.normal))
                dt = float(np.dot(ts.origin - to.origin, to.normal))
                if abs(dq + dt) > cfg.max_clearance_mm + 0.5:
                    return False

    return True


def _chirality_ok(
    q_list: list[FaceFeature],
    t_list: list[FaceFeature],
) -> bool:
    """
    For any triple with non-coplanar directions, verify handedness matches SO(3)
    with outward-normal sign inversions (det(-R) = -1 for 3 planes/freeform).
    """
    def _dir(f: FaceFeature) -> tuple[Optional[np.ndarray], int]:
        if f.surface_type in (SurfaceType.PLANE, SurfaceType.FREEFORM):
            return f.normal, -1
        if f.surface_type == SurfaceType.CONE:
            return f.axis, +1
        return None, 0

    n = len(q_list)
    if n < 3:
        return True
    last = n - 1
    vq_last, sq_last = _dir(q_list[last])
    vt_last, _ = _dir(t_list[last])
    if vq_last is None or vt_last is None:
        return True

    for i in range(last):
        vqi, sqi = _dir(q_list[i])
        vti, _ = _dir(t_list[i])
        if vqi is None or vti is None:
            continue
        for j in range(i + 1, last):
            vqj, sqj = _dir(q_list[j])
            vtj, _ = _dir(t_list[j])
            if vqj is None or vtj is None:
                continue
            det_q = float(np.linalg.det(np.column_stack([vqi, vqj, vq_last])))
            if abs(det_q) < 0.08:
                continue
            det_t = float(np.linalg.det(np.column_stack([vti, vtj, vt_last])))
            expected_sign = sqi * sqj * sq_last
            if det_q * det_t * expected_sign <= 0.0:
                return False
    return True


def _are_cosurface_siblings(t1: FaceFeature, t2: FaceFeature) -> bool:
    """
    Detect if two faces on the same part are seam-split / co-surface siblings
    of the exact same underlying geometric surface (e.g., two 180-deg half-cylinders).
    """
    if t1.part_id != t2.part_id or t1.global_id == t2.global_id:
        return False
    if t1.surface_type != t2.surface_type or t1.sense != t2.sense:
        return False
    if t2.local_face_id not in t1.adjacent_local_ids:
        return False

    st = t1.surface_type
    if st == SurfaceType.CYLINDER:
        if abs((t1.radius or 0.0) - (t2.radius or 0.0)) > 0.02:
            return False
        if abs(float(np.dot(t1.axis, t2.axis))) < 0.999:
            return False
        diff = t2.origin - t1.origin
        perp = diff - float(np.dot(diff, t1.axis)) * t1.axis
        return float(np.linalg.norm(perp)) <= 0.02

    if st == SurfaceType.CONE:
        if abs((t1.half_angle or 0.0) - (t2.half_angle or 0.0)) > 0.005:
            return False
        if float(np.dot(t1.axis, t2.axis)) < 0.999:
            return False
        return float(np.linalg.norm(t1.origin - t2.origin)) <= 0.05

    if st == SurfaceType.SPHERE:
        if abs((t1.radius or 0.0) - (t2.radius or 0.0)) > 0.02:
            return False
        return float(np.linalg.norm(t1.origin - t2.origin)) <= 0.05

    if st == SurfaceType.PLANE:
        if float(np.dot(t1.normal, t2.normal)) < 0.999:
            return False
        return abs(float(np.dot(t2.origin - t1.origin, t1.normal))) <= 0.02

    return False


def _expand_and_deduplicate_cosurface_siblings(
    results: list[MatchCandidate],
    target_parts: dict[int, PartData],
) -> list[MatchCandidate]:
    """
    Attach seam-split co-surface sibling faces on target parts to each match
    and deduplicate matches that cover the same unified logical surface set.
    """
    dedup: dict[frozenset[int], MatchCandidate] = {}
    for m in results:
        expanded_gids = list(m.target_global_face_ids)
        expanded_lids = list(m.target_local_face_ids)
        expanded_pids = list(m.target_part_ids)
        seen_gids = set(expanded_gids)

        for gid, lid, pid in zip(
            list(m.target_global_face_ids),
            list(m.target_local_face_ids),
            list(m.target_part_ids),
        ):
            part = target_parts.get(pid)
            if part is None or lid >= len(part.faces):
                continue
            tf = part.faces[lid]
            for adj_lid in tf.adjacent_local_ids:
                sib = part.faces[adj_lid]
                if sib.global_id not in seen_gids and _are_cosurface_siblings(tf, sib):
                    seen_gids.add(sib.global_id)
                    expanded_gids.append(sib.global_id)
                    expanded_lids.append(sib.local_face_id)
                    expanded_pids.append(sib.part_id)

        m.target_global_face_ids = expanded_gids
        m.target_local_face_ids = expanded_lids
        m.target_part_ids = expanded_pids
        key = frozenset(expanded_gids)
        if key not in dedup or m.score > dedup[key].score:
            dedup[key] = m

    return list(dedup.values())


def _search_correspondences_in_faces(
    query_part: PartData,
    query_faces: list[FaceFeature],
    candidate_pool: list[FaceFeature],
    target_parts: dict[int, PartData],
    cfg: MatingConfig,
    verified_sets: set[frozenset[int]],
    results: list[MatchCandidate],
) -> None:
    """Backtracking subgraph correspondence search across candidate_pool."""
    k = len(query_faces)
    if k == 0 or len(candidate_pool) < k:
        return

    # Build per-query candidate lists
    cands_per_q: list[list[FaceFeature]] = []
    for q in query_faces:
        cands = [t for t in candidate_pool if _single_face_compatible(q, t, cfg)]
        if not cands:
            return
        if len(cands) > cfg.max_candidates_per_face:
            # Prefer faces with closer area/radius when pool is huge
            cands.sort(key=lambda t: abs(math.log((t.area + 1e-3) / (q.area + 1e-3))))
            cands = cands[: cfg.max_candidates_per_face]
        cands_per_q.append(cands)

    # Order query faces most-constrained-first (fewest candidates, non-planar first)
    order = sorted(
        range(k),
        key=lambda idx: (
            len(cands_per_q[idx]),
            0 if query_faces[idx].surface_type != SurfaceType.PLANE else 1,
        ),
    )
    q_ordered = [query_faces[idx] for idx in order]
    c_ordered = [cands_per_q[idx] for idx in order]

    assigned_q: list[FaceFeature] = []
    assigned_t: list[FaceFeature] = []
    used_t_ids: set[int] = set()

    def backtrack(depth: int) -> None:
        if depth == k:
            key = frozenset(t.global_id for t in assigned_t)
            if key in verified_sets:
                return
            # Restore original query face order for clean reporting
            inv_map = {order[d]: assigned_t[d] for d in range(k)}
            pairs = [(query_faces[idx], inv_map[idx]) for idx in range(k)]
            match = solve_and_verify_match(
                pairs=pairs,
                target_parts=target_parts,
                cfg=cfg,
                match_id=len(results),
                query_part=query_part,
            )
            if match is not None:
                verified_sets.add(key)
                results.append(match)
            return

        q_cur = q_ordered[depth]
        for t_cand in c_ordered[depth]:
            if t_cand.global_id in used_t_ids:
                continue
            # Check pairwise compatibility with all already-assigned faces
            ok = True
            for prev_q, prev_t in zip(assigned_q, assigned_t):
                if not _pairwise_compatible(prev_q, q_cur, prev_t, t_cand, cfg):
                    ok = False
                    break
            if not ok:
                continue

            assigned_q.append(q_cur)
            assigned_t.append(t_cand)
            if _chirality_ok(assigned_q, assigned_t):
                used_t_ids.add(t_cand.global_id)
                backtrack(depth + 1)
                used_t_ids.remove(t_cand.global_id)
            assigned_q.pop()
            assigned_t.pop()

    backtrack(0)


def find_mating_instances(
    query_part: PartData,
    selected_face_ids: list[int],
    assembly: AssemblyData,
    config: Optional[MatingConfig] = None,
    exclude_part_ids: Optional[set[int]] = None,
    auto_exclude_self: bool = True,
) -> list[MatchCandidate]:
    """
    Find ALL places in `assembly` where `selected_face_ids` on `query_part`
    fit or mate (exact/interference or sliding/clearance), invariant to SE(3) pose.
    """
    cfg = config or MatingConfig()
    if not selected_face_ids:
        raise ValueError("selected_face_ids must not be empty.")

    n_faces = len(query_part.faces)
    for fid in selected_face_ids:
        if fid < 0 or fid >= n_faces:
            raise IndexError(f"Face ID {fid} out of range [0, {n_faces - 1}] for part '{query_part.name}'.")

    query_faces = [query_part.faces[fid] for fid in selected_face_ids]

    excluded = set(exclude_part_ids or ())
    if auto_exclude_self:
        for part in assembly.parts:
            if is_same_part_instance(query_part, part):
                excluded.add(part.part_id)

    search_parts = [p for p in assembly.parts if p.part_id not in excluded]
    target_parts = {p.part_id: p for p in search_parts}

    verified_sets: set[frozenset[int]] = set()
    results: list[MatchCandidate] = []

    # Pass 1: Search within each individual part (fast & covers single-part sockets/ways/bores)
    for part in search_parts:
        _search_correspondences_in_faces(
            query_part=query_part,
            query_faces=query_faces,
            candidate_pool=part.faces,
            target_parts=target_parts,
            cfg=cfg,
            verified_sets=verified_sets,
            results=results,
        )

    # Pass 2: Cross-part local clusters (for split housings, bolted rails, multi-part jaws)
    if cfg.allow_cross_part_matches and len(query_faces) >= 2 and len(search_parts) >= 2:
        q_bmin = np.min([q.bbox_min for q in query_faces], axis=0)
        q_bmax = np.max([q.bbox_max for q in query_faces], axis=0)
        q_diag = float(np.linalg.norm(q_bmax - q_bmin))
        cluster_radius = q_diag + cfg.max_cross_part_gap_mm

        for i, p_anchor in enumerate(search_parts):
            local_faces = list(p_anchor.faces)
            has_neighbor = False
            for j, p_other in enumerate(search_parts):
                if i == j:
                    continue
                d_min, _ = _bbox_min_max_dist(
                    p_anchor.bbox_min, p_anchor.bbox_max, p_other.bbox_min, p_other.bbox_max
                )
                if d_min <= cluster_radius:
                    local_faces.extend(p_other.faces)
                    has_neighbor = True
            if has_neighbor:
                _search_correspondences_in_faces(
                    query_part=query_part,
                    query_faces=query_faces,
                    candidate_pool=local_faces,
                    target_parts=target_parts,
                    cfg=cfg,
                    verified_sets=verified_sets,
                    results=results,
                )

    results = _expand_and_deduplicate_cosurface_siblings(results, target_parts)

    # Sort best score first, then deterministic by part_id / face_ids
    results.sort(key=lambda m: (-m.score, m.part_ids, m.target_global_face_ids))
    for idx, m in enumerate(results):
        m.match_id = idx
    return results
