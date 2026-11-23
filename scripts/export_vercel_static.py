"""Export all 9 scenarios, 3D B-Rep face meshes, analytic features, and VTK renders to public/ for Vercel deployment."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import numpy as np

from automate.features import extract_assembly_features, extract_part_features
from automate.matcher import find_mating_instances
from automate.models import FaceFeature, MatingConfig, PartData
from automate.step_io import is_same_part_instance, load_step_assembly, load_step_part
from automate.ui import INDEX_HTML, _face_to_mesh_dict, _shape_edges_to_segments
from OCP.BRepMesh import BRepMesh_IncrementalMesh


def _export_face_full(f: FaceFeature) -> dict:
    mesh = _face_to_mesh_dict(f.occt_face) or {"vertices": [], "indices": []}
    step = max(1, len(f.sample_points) // 25)
    sub_pts = np.round(f.sample_points[::step][:25], 3).tolist() if len(f.sample_points) else []
    sub_nrms = np.round(f.sample_normals[::step][:25], 4).tolist() if len(f.sample_normals) else []
    return {
        "global_id": f.global_id,
        "local_face_id": f.local_face_id,
        "part_id": f.part_id,
        "part_name": f.part_name,
        "surface_type": f.surface_type.value,
        "sense": f.sense,
        "area": round(float(f.area), 2),
        "centroid": [round(float(x), 3) for x in f.centroid],
        "bbox_min": [round(float(x), 3) for x in f.bbox_min],
        "bbox_max": [round(float(x), 3) for x in f.bbox_max],
        "normal": [round(float(x), 5) for x in f.normal] if f.normal is not None else None,
        "origin": [round(float(x), 4) for x in f.origin] if f.origin is not None else None,
        "axis": [round(float(x), 5) for x in f.axis] if f.axis is not None else None,
        "radius": round(float(f.radius), 4) if f.radius is not None else None,
        "r_min": round(float(f.r_min), 4) if f.r_min is not None else None,
        "r_max": round(float(f.r_max), 4) if f.r_max is not None else None,
        "half_angle": round(float(f.half_angle), 5) if f.half_angle is not None else None,
        "half_angle_deg": round(float(np.degrees(f.half_angle)), 2) if f.half_angle is not None else None,
        "adjacent_local_ids": list(f.adjacent_local_ids),
        "sample_points": sub_pts,
        "sample_normals": sub_nrms,
        "vertices": mesh["vertices"],
        "indices": mesh["indices"],
    }


def _export_part_full(part: PartData) -> dict:
    diag = float(np.linalg.norm(part.bbox_max - part.bbox_min))
    defl = max(0.04, diag * 0.005)
    BRepMesh_IncrementalMesh(part.shape, defl, False, 0.35, True)
    return {
        "part_id": part.part_id,
        "name": part.name,
        "volume": round(float(part.volume), 2),
        "bbox_min": [round(float(x), 3) for x in part.bbox_min],
        "bbox_max": [round(float(x), 3) for x in part.bbox_max],
        "faces": [_export_face_full(f) for f in part.faces],
        "edges": _shape_edges_to_segments(part.shape, defl),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    pub = root / "public"
    data_dir = pub / "data"
    out_pub_dir = pub / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_pub_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((root / "scenarios" / "manifest.json").read_text(encoding="utf-8"))
    (data_dir / "scenarios.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (pub / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    for sc in manifest:
        name = sc["name"]
        q_part = load_step_part(root / sc["part"])
        extract_part_features(q_part)
        asm = load_step_assembly(root / sc["assembly"])
        extract_assembly_features(asm)

        matches = find_mating_instances(q_part, sc["faces"], asm, MatingConfig())
        self_pids = [pt.part_id for pt in asm.parts if is_same_part_instance(q_part, pt)]

        png_name = f"{name}_render.png"
        src_png = root / "outputs" / png_name
        if src_png.exists():
            shutil.copy2(src_png, out_pub_dir / png_name)

        payload = {
            "scenario": name,
            "part_path": sc["part"],
            "assembly_path": sc["assembly"],
            "default_face_ids": sc["faces"],
            "selected_face_ids": sc["faces"],
            "self_part_ids": self_pids,
            "query_part": _export_part_full(q_part),
            "assembly_parts": [_export_part_full(pt) for pt in asm.parts],
            "default_matches": [m.to_dict() for m in matches],
            "matches": [m.to_dict() for m in matches],
            "render_url": f"/outputs/{png_name}",
        }
        (data_dir / f"{name}.json").write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        print(f"Exported {name}.json ({len(matches)} matches)")


if __name__ == "__main__":
    main()
