from __future__ import annotations

import math
from typing import Iterable
import numpy as np

from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import (
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Face
from OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
    IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
)
from OCP.gp import gp_Pnt2d

from .models import AssemblyData, FaceFeature, PartData, SurfaceType
from .step_io import get_face_map


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _angular_span(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    """Compute circular angular coverage [0, 2*pi] of points around an axis."""
    if len(points) < 3:
        return 0.0
    a = _normalize(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _normalize(np.cross(a, ref))
    e2 = _normalize(np.cross(a, e1))
    rel = points - origin
    x = rel @ e1
    y = rel @ e2
    r = np.hypot(x, y)
    valid = r > 1e-6
    if np.sum(valid) < 2:
        return 0.0
    angles = np.sort(np.arctan2(y[valid], x[valid]))
    diffs = np.diff(angles)
    wrap = (angles[0] + 2.0 * math.pi) - angles[-1]
    max_gap = float(max(np.max(diffs), wrap))
    return float(max(0.0, min(2.0 * math.pi, 2.0 * math.pi - max_gap)))


def _sample_face_uv(
    face: TopoDS_Face,
    adaptor: BRepAdaptor_Surface,
    grid_n: int = 9,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sample interior (points, outward_normals) strictly inside the trimmed B-Rep face."""
    umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    umin = max(umin, -1e4)
    umax = min(umax, 1e4)
    vmin = max(vmin, -1e4)
    vmax = min(vmax, 1e4)

    reversed_face = face.Orientation() == TopAbs_REVERSED
    clf = BRepClass_FaceClassifier()
    props = BRepLProp_SLProps(adaptor, 2, 1e-6)

    pts: list[np.ndarray] = []
    nrms: list[np.ndarray] = []
    mean_curvs: list[float] = []
    gauss_curvs: list[float] = []

    def try_uv(u: float, v: float) -> None:
        clf.Perform(face, gp_Pnt2d(u, v), 1e-6, True)
        state = clf.State()
        if state not in (TopAbs_IN, TopAbs_ON):
            return
        props.SetParameters(u, v)
        p = props.Value()
        if not props.IsNormalDefined():
            return
        n_dir = props.Normal()
        n = np.array([n_dir.X(), n_dir.Y(), n_dir.Z()], dtype=float)
        if reversed_face:
            n = -n
        pts.append(np.array([p.X(), p.Y(), p.Z()], dtype=float))
        nrms.append(_normalize(n))
        if props.IsCurvatureDefined():
            mc = float(props.MeanCurvature())
            gc = float(props.GaussianCurvature())
            if reversed_face:
                mc = -mc
            mean_curvs.append(mc)
            gauss_curvs.append(gc)

    for n_sub in (grid_n, 17):
        us = np.linspace(umin, umax, n_sub + 2)[1:-1]
        vs = np.linspace(vmin, vmax, n_sub + 2)[1:-1]
        for u in us:
            for v in vs:
                try_uv(float(u), float(v))
        if len(pts) >= 9:
            break

    if not pts:
        # Fallback on midpoint or boundary-inclusive grid
        for u in np.linspace(umin, umax, 5):
            for v in np.linspace(vmin, vmax, 5):
                props.SetParameters(float(u), float(v))
                if props.IsNormalDefined():
                    p = props.Value()
                    n_dir = props.Normal()
                    n = np.array([n_dir.X(), n_dir.Y(), n_dir.Z()], dtype=float)
                    if reversed_face:
                        n = -n
                    pts.append(np.array([p.X(), p.Y(), p.Z()], dtype=float))
                    nrms.append(_normalize(n))
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3)), 0.0, 0.0

    mc_val = float(np.median(mean_curvs)) if mean_curvs else 0.0
    gc_val = float(np.median(gauss_curvs)) if gauss_curvs else 0.0
    return np.array(pts, dtype=float), np.array(nrms, dtype=float), mc_val, gc_val


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues desc, eigenvectors rows) for 3D point cloud."""
    if len(points) < 3:
        return np.zeros(3), np.eye(3)
    centered = points - np.mean(points, axis=0)
    cov = (centered.T @ centered) / max(1, len(points))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order].T  # rows are principal directions
    if np.linalg.det(eigvecs) < 0:
        eigvecs[2] = -eigvecs[2]
    return eigvals, eigvecs


def extract_face_feature(
    face: TopoDS_Face,
    global_id: int,
    part_id: int,
    part_name: str,
    local_face_id: int,
) -> FaceFeature:
    """Extract analytic/freeform geometry, outward normals, and samples from a TopoDS_Face."""
    gprops = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, gprops)
    area = abs(float(gprops.Mass()))
    c_pnt = gprops.CentreOfMass()
    centroid = np.array([c_pnt.X(), c_pnt.Y(), c_pnt.Z()], dtype=float)

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(face, box, False, False)
    if box.IsVoid():
        bbox_min = centroid.copy()
        bbox_max = centroid.copy()
    else:
        cmin = box.CornerMin()
        cmax = box.CornerMax()
        bbox_min = np.array([cmin.X(), cmin.Y(), cmin.Z()], dtype=float)
        bbox_max = np.array([cmax.X(), cmax.Y(), cmax.Z()], dtype=float)

    adaptor = BRepAdaptor_Surface(face, True)
    stype = adaptor.GetType()
    pts, nrms, mean_k, gauss_k = _sample_face_uv(face, adaptor)

    if len(pts) == 0:
        pts = centroid.reshape(1, 3)

    reversed_face = face.Orientation() == TopAbs_REVERSED

    if stype == GeomAbs_Plane:
        pln = adaptor.Plane()
        n_dir = pln.Axis().Direction()
        normal = np.array([n_dir.X(), n_dir.Y(), n_dir.Z()], dtype=float)
        if reversed_face:
            normal = -normal
        normal = _normalize(normal)
        loc = pln.Location()
        loc_arr = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
        origin = centroid - float(np.dot(centroid - loc_arr, normal)) * normal
        _, eigvecs = _pca_frame(pts)
        proj = pts - np.mean(pts, axis=0)
        # Project onto tangent plane orthonormal basis
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = _normalize(np.cross(normal, ref))
        t2 = _normalize(np.cross(normal, t1))
        pts_2d = np.column_stack([proj @ t1, proj @ t2])
        _, eig2 = _pca_frame(np.column_stack([pts_2d, np.zeros(len(pts_2d))]))
        rot_2d = pts_2d @ eig2[:2, :2].T
        ext_u = float(np.max(rot_2d[:, 0]) - np.min(rot_2d[:, 0]))
        ext_v = float(np.max(rot_2d[:, 1]) - np.min(rot_2d[:, 1]))

        return FaceFeature(
            global_id=global_id,
            part_id=part_id,
            part_name=part_name,
            local_face_id=local_face_id,
            surface_type=SurfaceType.PLANE,
            area=area,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            sense=0,
            normal=normal,
            origin=origin,
            extent_u=ext_u,
            extent_v=ext_v,
            mean_curvature=0.0,
            gaussian_curvature=0.0,
            pca_axes=eigvecs,
            sample_points=pts,
            sample_normals=np.tile(normal, (len(pts), 1)) if len(pts) else nrms,
            occt_face=face,
        )

    if stype == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        ax_dir = cyl.Axis().Direction()
        axis = _normalize(np.array([ax_dir.X(), ax_dir.Y(), ax_dir.Z()], dtype=float))
        loc = cyl.Location()
        origin = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
        radius = float(cyl.Radius())

        # Determine convex (+1) vs concave (-1) from outward sample normals
        sense = 1
        if len(nrms) > 0:
            rel = pts - origin
            radial = rel - np.outer(rel @ axis, axis)
            dots = np.sum(radial * nrms, axis=1)
            sense = 1 if float(np.median(dots)) >= 0.0 else -1

        ax_coords = (pts - origin) @ axis
        axial_span = float(np.max(ax_coords) - np.min(ax_coords))
        ang_span = _angular_span(pts, origin, axis)

        return FaceFeature(
            global_id=global_id,
            part_id=part_id,
            part_name=part_name,
            local_face_id=local_face_id,
            surface_type=SurfaceType.CYLINDER,
            area=area,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            sense=sense,
            origin=origin,
            axis=axis,
            radius=radius,
            axial_span=axial_span,
            angular_span=ang_span,
            mean_curvature=mean_k,
            gaussian_curvature=0.0,
            sample_points=pts,
            sample_normals=nrms,
            occt_face=face,
        )

    if stype == GeomAbs_Cone:
        cone = adaptor.Cone()
        apex_pnt = cone.Apex()
        apex = np.array([apex_pnt.X(), apex_pnt.Y(), apex_pnt.Z()], dtype=float)
        ax_dir = cone.Axis().Direction()
        raw_axis = _normalize(np.array([ax_dir.X(), ax_dir.Y(), ax_dir.Z()], dtype=float))
        half_angle = abs(float(cone.SemiAngle()))

        # Canonicalize axis to point from apex toward the widening side of the face
        proj_sum = float(np.mean((pts - apex) @ raw_axis))
        axis = raw_axis if proj_sum >= 0.0 else -raw_axis

        rel = pts - apex
        ax_coords = rel @ axis
        radial_vec = rel - np.outer(ax_coords, axis)
        radii = np.linalg.norm(radial_vec, axis=1)
        r_min = float(np.min(radii))
        r_max = float(np.max(radii))
        r_mean = 0.5 * (r_min + r_max)
        axial_span = float(np.max(ax_coords) - np.min(ax_coords))
        ang_span = _angular_span(pts, apex, axis)

        sense = 1
        if len(nrms) > 0:
            dots = np.sum(radial_vec * nrms, axis=1)
            sense = 1 if float(np.median(dots)) >= 0.0 else -1

        return FaceFeature(
            global_id=global_id,
            part_id=part_id,
            part_name=part_name,
            local_face_id=local_face_id,
            surface_type=SurfaceType.CONE,
            area=area,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            sense=sense,
            origin=apex,
            axis=axis,
            radius=r_mean,
            r_min=r_min,
            r_max=r_max,
            half_angle=half_angle,
            axial_span=axial_span,
            angular_span=ang_span,
            mean_curvature=mean_k,
            gaussian_curvature=0.0,
            sample_points=pts,
            sample_normals=nrms,
            occt_face=face,
        )

    if stype == GeomAbs_Sphere:
        sph = adaptor.Sphere()
        loc = sph.Location()
        center = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
        radius = float(sph.Radius())
        sense = 1
        if len(nrms) > 0:
            dots = np.sum((pts - center) * nrms, axis=1)
            sense = 1 if float(np.median(dots)) >= 0.0 else -1

        return FaceFeature(
            global_id=global_id,
            part_id=part_id,
            part_name=part_name,
            local_face_id=local_face_id,
            surface_type=SurfaceType.SPHERE,
            area=area,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            sense=sense,
            origin=center,
            radius=radius,
            mean_curvature=mean_k,
            gaussian_curvature=gauss_k,
            sample_points=pts,
            sample_normals=nrms,
            occt_face=face,
        )

    if stype == GeomAbs_Torus:
        tor = adaptor.Torus()
        loc = tor.Location()
        center = np.array([loc.X(), loc.Y(), loc.Z()], dtype=float)
        ax_dir = tor.Axis().Direction()
        axis = _normalize(np.array([ax_dir.X(), ax_dir.Y(), ax_dir.Z()], dtype=float))
        maj_r = float(tor.MajorRadius())
        min_r = float(tor.MinorRadius())
        sense = 1 if mean_k >= 0.0 else -1
        return FaceFeature(
            global_id=global_id,
            part_id=part_id,
            part_name=part_name,
            local_face_id=local_face_id,
            surface_type=SurfaceType.TORUS,
            area=area,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            sense=sense,
            origin=center,
            axis=axis,
            radius=min_r,
            major_radius=maj_r,
            minor_radius=min_r,
            mean_curvature=mean_k,
            gaussian_curvature=gauss_k,
            sample_points=pts,
            sample_normals=nrms,
            occt_face=face,
        )

    # Freeform (BSpline, Bezier, Revolution, Extrusion, Other)
    mean_n = _normalize(np.mean(nrms, axis=0)) if len(nrms) > 0 else np.array([0.0, 0.0, 1.0])
    _, eigvecs = _pca_frame(pts)
    sense = 1 if mean_k > 1e-4 else (-1 if mean_k < -1e-4 else 0)
    return FaceFeature(
        global_id=global_id,
        part_id=part_id,
        part_name=part_name,
        local_face_id=local_face_id,
        surface_type=SurfaceType.FREEFORM,
        area=area,
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        sense=sense,
        normal=mean_n,
        origin=centroid,
        mean_curvature=mean_k,
        gaussian_curvature=gauss_k,
        pca_axes=eigvecs,
        sample_points=pts,
        sample_normals=nrms,
        occt_face=face,
    )


def _compute_face_adjacency(part: PartData, fmap: IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher) -> None:
    edge_face_map = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapesAndAncestors_s(part.shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)
    adj: dict[int, set[int]] = {i: set() for i in range(len(part.faces))}
    for e_idx in range(1, edge_face_map.Extent() + 1):
        face_list = edge_face_map.FindFromIndex(e_idx)
        local_ids: list[int] = []
        for f_shape in face_list:
            f_idx = fmap.FindIndex(f_shape)  # 1-based
            if f_idx >= 1:
                local_ids.append(f_idx - 1)
        for i in range(len(local_ids)):
            for j in range(i + 1, len(local_ids)):
                a, b = local_ids[i], local_ids[j]
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
    for idx, feat in enumerate(part.faces):
        feat.adjacent_local_ids = tuple(sorted(adj[idx]))


def extract_part_features(part: PartData, start_global_id: int = 0) -> int:
    """Extract FaceFeature descriptors and adjacency for all faces on a PartData."""
    fmap = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(part.shape, TopAbs_FACE, fmap)
    part.faces = []
    gid = start_global_id
    for i in range(1, fmap.Extent() + 1):
        face = TopoDS.Face(fmap.FindKey(i))
        feat = extract_face_feature(
            face=face,
            global_id=gid,
            part_id=part.part_id,
            part_name=part.name,
            local_face_id=i - 1,
        )
        part.faces.append(feat)
        gid += 1
    _compute_face_adjacency(part, fmap)
    return gid


def extract_assembly_features(assembly: AssemblyData) -> AssemblyData:
    """Extract features for every part in an AssemblyData."""
    gid = 0
    for part in assembly.parts:
        gid = extract_part_features(part, start_global_id=gid)
    return assembly
