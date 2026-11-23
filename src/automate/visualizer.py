from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import vtk
from vtk.util import numpy_support

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GCPnts import GCPnts_UniformDeflection
from OCP.TopAbs import TopAbs_EDGE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher

from .models import AssemblyData, FitType, MatchCandidate, PartData
from .step_io import is_same_part_instance


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _face_to_vtk_polydata(
    face,
    deflection: float = 0.25,
    transform_4x4: Optional[np.ndarray] = None,
) -> Optional[vtk.vtkPolyData]:
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbNodes() == 0 or tri.NbTriangles() == 0:
        return None

    trsf = loc.Transformation()
    is_identity = loc.IsIdentity()
    rev = face.Orientation() == TopAbs_REVERSED

    n_nodes = tri.NbNodes()
    pts_np = np.empty((n_nodes, 3), dtype=np.float64)
    for i in range(1, n_nodes + 1):
        p = tri.Node(i)
        if not is_identity:
            p = p.Transformed(trsf)
        pts_np[i - 1, 0] = p.X()
        pts_np[i - 1, 1] = p.Y()
        pts_np[i - 1, 2] = p.Z()

    if transform_4x4 is not None:
        R = transform_4x4[:3, :3]
        t = transform_4x4[:3, 3]
        pts_np = (pts_np @ R.T) + t

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(pts_np, deep=True))

    cells = vtk.vtkCellArray()
    n_tris = tri.NbTriangles()
    for i in range(1, n_tris + 1):
        a, b, c = tri.Triangle(i).Get()
        if rev:
            b, c = c, b
        cells.InsertNextCell(3)
        cells.InsertCellPoint(a - 1)
        cells.InsertCellPoint(b - 1)
        cells.InsertCellPoint(c - 1)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetPolys(cells)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.SplittingOn()
    normals.SetFeatureAngle(35.0)
    normals.Update()
    return normals.GetOutput()


def _shape_edges_to_vtk_polydata(
    shape,
    deflection: float = 0.25,
    transform_4x4: Optional[np.ndarray] = None,
) -> Optional[vtk.vtkPolyData]:
    emap = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, emap)
    if emap.Extent() == 0:
        return None

    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    pt_idx = 0

    R = transform_4x4[:3, :3] if transform_4x4 is not None else None
    t = transform_4x4[:3, 3] if transform_4x4 is not None else None

    for i in range(1, emap.Extent() + 1):
        edge = TopoDS.Edge(emap.FindKey(i))
        try:
            curve = BRepAdaptor_Curve(edge)
            disc = GCPnts_UniformDeflection(curve, deflection)
            if not disc.IsDone() or disc.NbPoints() < 2:
                continue
            n_p = disc.NbPoints()
            lines.InsertNextCell(n_p)
            for j in range(1, n_p + 1):
                p = disc.Value(j)
                if R is not None and t is not None:
                    pw = R @ np.array([p.X(), p.Y(), p.Z()], dtype=float) + t
                    pts.InsertNextPoint(float(pw[0]), float(pw[1]), float(pw[2]))
                else:
                    pts.InsertNextPoint(p.X(), p.Y(), p.Z())
                lines.InsertCellPoint(pt_idx)
                pt_idx += 1
        except Exception:
            continue

    if pt_idx == 0:
        return None
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    return poly


def _add_poly_actor(
    renderer: vtk.vtkRenderer,
    poly: vtk.vtkPolyData,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    specular: float = 0.25,
) -> None:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetSpecular(specular)
    prop.SetSpecularPower(22.0)
    prop.SetAmbient(0.20)
    prop.SetDiffuse(0.76)
    renderer.AddActor(actor)


def _add_edge_actor(
    renderer: vtk.vtkRenderer,
    poly: vtk.vtkPolyData,
    color: tuple[float, float, float] = (0.08, 0.08, 0.10),
    opacity: float = 0.55,
    line_width: float = 1.0,
) -> None:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetLineWidth(line_width)
    renderer.AddActor(actor)


def _render_scene_to_image(
    renderer: vtk.vtkRenderer,
    width: int,
    height: int,
    azimuth_deg: float = 38.0,
    elevation_deg: float = 26.0,
) -> Image.Image:
    renderer.SetBackground(0.14, 0.15, 0.17)
    renderer.UseDepthPeelingOn()
    renderer.SetMaximumNumberOfPeels(10)
    renderer.SetOcclusionRatio(0.04)

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetAlphaBitPlanes(1)
    rw.SetMultiSamples(0)
    rw.SetSize(width, height)
    rw.AddRenderer(renderer)

    cam = renderer.GetActiveCamera()
    renderer.ResetCamera()
    cam.Azimuth(azimuth_deg)
    cam.Elevation(elevation_deg)
    cam.OrthogonalizeViewUp()
    renderer.ResetCameraClippingRange()
    cam.Zoom(1.12)

    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    vtk_img = w2i.GetOutput()
    dims = vtk_img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    arr = arr.reshape(dims[1], dims[0], 3)
    arr = np.flipud(arr)
    rw.Finalize()
    return Image.fromarray(arr, mode="RGB")


def _render_query_panel(
    query_part: PartData,
    selected_face_ids: list[int],
    width: int,
    height: int,
) -> Image.Image:
    diag = float(np.linalg.norm(query_part.bbox_max - query_part.bbox_min))
    defl = max(0.03, diag * 0.004)
    BRepMesh_IncrementalMesh(query_part.shape, defl, False, 0.35, True)

    ren = vtk.vtkRenderer()
    sel_set = set(selected_face_ids)
    for f in query_part.faces:
        poly = _face_to_vtk_polydata(f.occt_face, defl)
        if poly is None:
            continue
        if f.local_face_id in sel_set:
            _add_poly_actor(ren, poly, color=(0.18, 0.88, 0.38), opacity=0.98)
        else:
            _add_poly_actor(ren, poly, color=(0.56, 0.59, 0.64), opacity=0.45)

    edge_poly = _shape_edges_to_vtk_polydata(query_part.shape, defl)
    if edge_poly is not None:
        _add_edge_actor(ren, edge_poly, color=(0.05, 0.05, 0.07), opacity=0.65, line_width=1.1)

    return _render_scene_to_image(ren, width, height)


def _render_assembly_panel(
    assembly: AssemblyData,
    matches: list[MatchCandidate],
    width: int,
    height: int,
) -> Image.Image:
    all_bmin = np.min([p.bbox_min for p in assembly.parts], axis=0)
    all_bmax = np.max([p.bbox_max for p in assembly.parts], axis=0)
    diag = float(np.linalg.norm(all_bmax - all_bmin))
    defl = max(0.05, diag * 0.003)

    matched_gids: dict[int, FitType] = {}
    matched_pids: set[int] = set()
    for m in matches:
        for pid in m.part_ids:
            matched_pids.add(pid)
        for gid in m.target_global_face_ids:
            matched_gids[gid] = m.fit_type

    ren = vtk.vtkRenderer()
    for part in assembly.parts:
        BRepMesh_IncrementalMesh(part.shape, defl, False, 0.35, True)
        is_matched_part = part.part_id in matched_pids
        for f in part.faces:
            poly = _face_to_vtk_polydata(f.occt_face, defl)
            if poly is None:
                continue
            if f.global_id in matched_gids:
                fit = matched_gids[f.global_id]
                if fit == FitType.SLIDING_CLEARANCE_FIT:
                    col = (0.14, 0.84, 0.92)
                elif fit == FitType.INTERFERENCE_FIT:
                    col = (0.98, 0.65, 0.15)
                else:
                    col = (0.18, 0.88, 0.38)
                _add_poly_actor(ren, poly, color=col, opacity=1.0)
            elif is_matched_part:
                _add_poly_actor(ren, poly, color=(0.35, 0.72, 0.45), opacity=0.30)
            else:
                _add_poly_actor(ren, poly, color=(0.56, 0.59, 0.63), opacity=0.20)

        edge_poly = _shape_edges_to_vtk_polydata(part.shape, defl)
        if edge_poly is not None:
            _add_edge_actor(
                ren,
                edge_poly,
                color=(0.08, 0.09, 0.11) if is_matched_part else (0.12, 0.13, 0.15),
                opacity=0.65 if is_matched_part else 0.28,
                line_width=1.2 if is_matched_part else 0.8,
            )

    return _render_scene_to_image(ren, width, height)


def _render_mated_verification_panel(
    query_part: PartData,
    selected_face_ids: list[int],
    assembly: AssemblyData,
    matches: list[MatchCandidate],
    width: int,
    height: int,
) -> Image.Image:
    """Render the assembly with the query part transformed by T into each matched location."""
    all_bmin = np.min([p.bbox_min for p in assembly.parts], axis=0)
    all_bmax = np.max([p.bbox_max for p in assembly.parts], axis=0)
    diag = float(np.linalg.norm(all_bmax - all_bmin))
    defl = max(0.05, diag * 0.003)

    matched_pids = {pid for m in matches for pid in m.part_ids}
    ren = vtk.vtkRenderer()

    # Render matched assembly target parts (and hide un-mated self copy of query_part for clarity)
    for part in assembly.parts:
        if is_same_part_instance(query_part, part):
            continue
        BRepMesh_IncrementalMesh(part.shape, defl, False, 0.35, True)
        is_matched = part.part_id in matched_pids
        for f in part.faces:
            poly = _face_to_vtk_polydata(f.occt_face, defl)
            if poly is None:
                continue
            _add_poly_actor(
                ren,
                poly,
                color=(0.45, 0.55, 0.62) if is_matched else (0.50, 0.52, 0.56),
                opacity=0.28 if is_matched else 0.14,
            )
        edge_poly = _shape_edges_to_vtk_polydata(part.shape, defl)
        if edge_poly is not None:
            _add_edge_actor(ren, edge_poly, color=(0.10, 0.12, 0.14), opacity=0.35, line_width=0.9)

    # Overlay the query part transformed by each match's SE(3) matrix T
    q_defl = max(0.03, float(np.linalg.norm(query_part.bbox_max - query_part.bbox_min)) * 0.004)
    BRepMesh_IncrementalMesh(query_part.shape, q_defl, False, 0.35, True)
    sel_set = set(selected_face_ids)

    for m in matches:
        T = m.transform_4x4
        for f in query_part.faces:
            poly = _face_to_vtk_polydata(f.occt_face, q_defl, transform_4x4=T)
            if poly is None:
                continue
            if f.local_face_id in sel_set:
                _add_poly_actor(ren, poly, color=(0.18, 0.92, 0.40), opacity=0.98)
            else:
                _add_poly_actor(ren, poly, color=(0.95, 0.76, 0.22), opacity=0.82)
        q_edges = _shape_edges_to_vtk_polydata(query_part.shape, q_defl, transform_4x4=T)
        if q_edges is not None:
            _add_edge_actor(ren, q_edges, color=(0.15, 0.12, 0.05), opacity=0.75, line_width=1.2)

    return _render_scene_to_image(ren, width, height)


def render_mating_results(
    query_part: PartData,
    selected_face_ids: list[int],
    assembly: AssemblyData,
    matches: list[MatchCandidate],
    output_path: str | Path,
    title: str = "Auto-Mating Result",
) -> Path:
    """
    Render a 3-panel PNG showing:
      - Left: Query part with selected faces highlighted in green
      - Center: Assembly with all matched target faces highlighted (Green=Exact, Cyan=Sliding/Clearance)
      - Right: Mated SE(3) verification view showing query part placed into each matched location
      - Bottom: Summary table of every match instance
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    w1, w2, w3, panel_h = 440, 620, 620, 580
    q_img = _render_query_panel(query_part, selected_face_ids, w1, panel_h)
    a_img = _render_assembly_panel(assembly, matches, w2, panel_h)
    m_img = _render_mated_verification_panel(query_part, selected_face_ids, assembly, matches, w3, panel_h)

    header_h = 48
    row_h = 24
    max_rows = min(len(matches), 10)
    footer_h = 56 + max(1, max_rows) * row_h

    total_w = w1 + w2 + w3
    total_h = header_h + panel_h + footer_h
    canvas = Image.new("RGB", (total_w, total_h), (24, 26, 30))
    canvas.paste(q_img, (0, header_h))
    canvas.paste(a_img, (w1, header_h))
    canvas.paste(m_img, (w1 + w2, header_h))

    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(15)
    font_row = _load_font(13)

    # Header bar
    draw.rectangle([0, 0, total_w, header_h], fill=(18, 20, 24))
    draw.text(
        (16, 15),
        f"1. Query: {query_part.name} (faces {selected_face_ids})",
        fill=(235, 240, 245),
        font=font_title,
    )
    draw.text(
        (w1 + 16, 15),
        f"2. Target Face Matches: {len(matches)} [Green=Exact, Cyan=Sliding/Clearance]",
        fill=(145, 235, 175),
        font=font_title,
    )
    draw.text(
        (w1 + w2 + 16, 15),
        "3. SE(3) Mated Pose Verification [Gold=Placed Query Part]",
        fill=(245, 210, 115),
        font=font_title,
    )
    draw.line([(w1, 0), (w1, header_h + panel_h)], fill=(65, 70, 78), width=2)
    draw.line([(w1 + w2, 0), (w1 + w2, header_h + panel_h)], fill=(65, 70, 78), width=2)

    # Footer summary
    y0 = header_h + panel_h + 10
    draw.text((16, y0), f"Scenario '{title}' — Matched Instances:", fill=(220, 225, 230), font=font_title)
    if not matches:
        draw.text((24, y0 + 26), "No mating instances found in assembly.", fill=(240, 130, 130), font=font_row)
    else:
        for idx, m in enumerate(matches[:max_rows]):
            y = y0 + 26 + idx * row_h
            parts_str = ", ".join(m.part_names)
            opp_str = f", opp_clear={m.opposing_clearance_mm:.3f}mm" if m.opposing_clearance_mm is not None else ""
            line = (
                f"[{m.match_id}] Part(s): {parts_str:24s} | TargetFaces: {str(m.target_local_face_ids):16s} | "
                f"{m.fit_type.value:21s} ({m.kinematic_pair.value:15s} T:{m.dof_translation} R:{m.dof_rotation}) | "
                f"gap={m.mean_gap_mm:.3f}mm{opp_str} | overlap={m.overlap_score:.2f} | score={m.score:.3f}"
            )
            col = (
                (110, 235, 245)
                if m.fit_type == FitType.SLIDING_CLEARANCE_FIT
                else ((250, 190, 90) if m.fit_type == FitType.INTERFERENCE_FIT else (115, 240, 145))
            )
            draw.text((20, y), line, fill=col, font=font_row)
        if len(matches) > max_rows:
            draw.text(
                (20, y0 + 26 + max_rows * row_h),
                f"... and {len(matches) - max_rows} more matches",
                fill=(180, 185, 190),
                font=font_row,
            )

    canvas.save(out)
    return out
