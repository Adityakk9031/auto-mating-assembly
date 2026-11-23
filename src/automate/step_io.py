from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TDF import TDF_Label
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher, Sequence_TDF_Label

from .models import AssemblyData, PartData


def _get_label_name(label: TDF_Label, default: str) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        name = attr.Get().ToExtString().strip()
        if name:
            return name
    return default


def _shape_bbox_and_volume(shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray, float]:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    if box.IsVoid():
        return np.zeros(3), np.zeros(3), 0.0
    cmin = box.CornerMin()
    cmax = box.CornerMax()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    vol = abs(float(props.Mass()))
    return (
        np.array([cmin.X(), cmin.Y(), cmin.Z()], dtype=float),
        np.array([cmax.X(), cmax.Y(), cmax.Z()], dtype=float),
        vol,
    )


def _split_into_bodies(shape: TopoDS_Shape, base_name: str) -> list[tuple[str, TopoDS_Shape]]:
    """Split a compound shape into individual solids (or free shells if no solids)."""
    solid_map = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, solid_map)
    n_solids = solid_map.Extent()
    if n_solids == 1:
        return [(base_name, solid_map.FindKey(1))]
    if n_solids > 1:
        bodies = []
        for i in range(1, n_solids + 1):
            bodies.append((f"{base_name}#{i}", solid_map.FindKey(i)))
        return bodies

    shell_map = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_SHELL, shell_map)
    n_shells = shell_map.Extent()
    if n_shells == 1:
        return [(base_name, shell_map.FindKey(1))]
    if n_shells > 1:
        return [(f"{base_name}#{i}", shell_map.FindKey(i)) for i in range(1, n_shells + 1)]

    face_map = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    if face_map.Extent() > 0:
        return [(base_name, shape)]
    return []


def _apply_location(shape: TopoDS_Shape, loc: TopLoc_Location) -> TopoDS_Shape:
    if loc.IsIdentity():
        return shape
    trsf = loc.Transformation()
    builder = BRepBuilderAPI_Transform(shape, trsf, True)
    return builder.Shape()


def _collect_xcaf_parts(
    shape_tool,
    label: TDF_Label,
    parent_loc: TopLoc_Location,
    inherited_name: Optional[str],
    out: list[tuple[str, TopoDS_Shape]],
) -> None:
    loc = parent_loc.Multiplied(shape_tool.GetLocation_s(label))
    name = _get_label_name(label, inherited_name or "Part")

    target_label = label
    if shape_tool.IsReference_s(label):
        ref_label = TDF_Label()
        if shape_tool.GetReferredShape_s(label, ref_label):
            ref_name = _get_label_name(ref_label, "")
            if ref_name and (name == "Part" or name.startswith("NAUO") or name.startswith("=>")):
                name = ref_name
            target_label = ref_label

    if shape_tool.IsAssembly_s(target_label):
        comps = Sequence_TDF_Label()
        shape_tool.GetComponents_s(target_label, comps, False)
        for i in range(1, comps.Length() + 1):
            _collect_xcaf_parts(shape_tool, comps.Value(i), loc, name, out)
        return

    shape = shape_tool.GetShape_s(target_label)
    if shape is None or shape.IsNull():
        return
    world_shape = _apply_location(shape, loc)
    out.extend(_split_into_bodies(world_shape, name))


def load_step_assembly(path: str | Path) -> AssemblyData:
    """Load a STEP assembly into a list of world-transformed PartData bodies."""
    step_path = Path(path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    raw_parts: list[tuple[str, TopoDS_Shape]] = []

    # Primary: XCAF reader preserves assembly hierarchy, instance locations, and part names
    try:
        app = XCAFApp_Application.GetApplication_s()
        doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
        app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
        caf_reader = STEPCAFControl_Reader()
        caf_reader.SetNameMode(True)
        status = caf_reader.ReadFile(str(step_path))
        if status == IFSelect_RetDone and caf_reader.Transfer(doc):
            shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
            free_labels = Sequence_TDF_Label()
            shape_tool.GetFreeShapes(free_labels)
            for i in range(1, free_labels.Length() + 1):
                _collect_xcaf_parts(
                    shape_tool,
                    free_labels.Value(i),
                    TopLoc_Location(),
                    None,
                    raw_parts,
                )
    except Exception:
        raw_parts = []

    # Fallback: plain STEPControl_Reader
    if not raw_parts:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to read STEP file: {step_path}")
        reader.TransferRoots()
        shape = reader.OneShape()
        raw_parts = _split_into_bodies(shape, step_path.stem)

    # Disambiguate duplicate names if XCAF gave identical prototype names to instances
    name_counts: dict[str, int] = {}
    parts: list[PartData] = []
    for idx, (name, shape) in enumerate(raw_parts):
        count = name_counts.get(name, 0) + 1
        name_counts[name] = count
        final_name = name if count == 1 else f"{name}_{count}"
        bmin, bmax, vol = _shape_bbox_and_volume(shape)
        parts.append(
            PartData(
                part_id=idx,
                name=final_name,
                shape=shape,
                bbox_min=bmin,
                bbox_max=bmax,
                volume=vol,
            )
        )

    return AssemblyData(source_path=str(step_path), parts=parts)


def load_step_part(path: str | Path, part_name: Optional[str] = None) -> PartData:
    """Load a single-part STEP file preserving exact deterministic face ordering."""
    step_path = Path(path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    bmin, bmax, vol = _shape_bbox_and_volume(shape)
    return PartData(
        part_id=-1,
        name=part_name or step_path.stem,
        shape=shape,
        bbox_min=bmin,
        bbox_max=bmax,
        volume=vol,
    )


def get_face_map(shape: TopoDS_Shape) -> list:
    """Return ordered list of unique TopoDS_Face objects (0-based index)."""
    fmap = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
    return [fmap.FindKey(i) for i in range(1, fmap.Extent() + 1)]


def is_same_part_instance(
    query_part: PartData,
    target_part: PartData,
    pos_tol_mm: float = 0.05,
    vol_rel_tol: float = 0.005,
) -> bool:
    """Detect if target_part is the exact same body at the exact same location as query_part."""
    if len(query_part.faces) != len(target_part.faces):
        return False
    if query_part.volume > 1e-6:
        if abs(query_part.volume - target_part.volume) / query_part.volume > vol_rel_tol:
            return False
    if np.linalg.norm(query_part.bbox_min - target_part.bbox_min) > pos_tol_mm:
        return False
    if np.linalg.norm(query_part.bbox_max - target_part.bbox_max) > pos_tol_mm:
        return False
    if not query_part.faces:
        return False
    q_centroids = np.sort(np.array([f.centroid for f in query_part.faces]), axis=0)
    t_centroids = np.sort(np.array([f.centroid for f in target_part.faces]), axis=0)
    return float(np.max(np.linalg.norm(q_centroids - t_centroids, axis=1))) < pos_tol_mm
