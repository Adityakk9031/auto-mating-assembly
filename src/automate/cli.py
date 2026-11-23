from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import extract_assembly_features, extract_part_features
from .matcher import find_mating_instances
from .models import MatingConfig
from .step_io import load_step_assembly, load_step_part
from .visualizer import render_mating_results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="automate",
        description="SE(3)-invariant B-Rep auto-mating in STEP assemblies.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("match", help="Find all mating instances for selected part faces in an assembly.")
    m.add_argument("--part", required=True, help="Path to single-part STEP file.")
    m.add_argument(
        "--faces",
        required=True,
        help="Comma-separated face IDs on the part (0-based by default; see --one-based).",
    )
    m.add_argument("--assembly", required=True, help="Path to assembly STEP file.")
    m.add_argument("--out-dir", default="outputs", help="Directory for JSON report and PNG render.")
    m.add_argument("--name", default=None, help="Scenario name for output files.")
    m.add_argument("--one-based", action="store_true", help="Interpret --faces as 1-based OCCT face IDs.")
    m.add_argument("--include-self", action="store_true", help="Allow matching against identical self-instance in assembly.")
    m.add_argument("--exact-tol", type=float, default=0.08, help="Gap threshold (mm) separating EXACT_FIT from SLIDING_CLEARANCE_FIT.")
    m.add_argument("--max-clearance", type=float, default=2.5, help="Maximum clearance gap (mm) for sliding/clearance mates.")
    m.add_argument("--no-render", action="store_true", help="Skip PNG rendering.")

    insp = sub.add_parser("inspect", help="List face IDs, surface types, areas, and centroids of a STEP file.")
    insp.add_argument("--step", required=True, help="Path to STEP file.")

    ui_p = sub.add_parser("ui", help="Launch the interactive 3D Web UI for exploring scenarios and auto-mating.")
    ui_p.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1).")
    ui_p.add_argument("--port", type=int, default=8050, help="Port to serve UI on (default: 8050).")
    return p


def cmd_inspect(step_path: str) -> None:
    part = load_step_part(step_path)
    extract_part_features(part)
    print(f"Part '{part.name}' | {len(part.faces)} faces | volume={part.volume:.2f}")
    for f in part.faces:
        extra = ""
        if f.radius is not None:
            extra += f" r={f.radius:.3f}"
        if f.half_angle is not None:
            extra += f" half_angle={f.half_angle * 180.0 / 3.14159265:.2f}deg"
        if f.normal is not None:
            extra += f" n=({f.normal[0]:.2f},{f.normal[1]:.2f},{f.normal[2]:.2f})"
        print(
            f"  face {f.local_face_id:3d}: {f.surface_type.value:8s} sense={f.sense:+d} "
            f"area={f.area:8.2f} centroid=({f.centroid[0]:.2f},{f.centroid[1]:.2f},{f.centroid[2]:.2f}){extra}"
        )


def cmd_match(args: argparse.Namespace) -> int:
    raw_ids = [int(x.strip()) for x in args.faces.split(",") if x.strip()]
    face_ids = [fid - 1 for fid in raw_ids] if args.one_based else raw_ids

    part = load_step_part(args.part)
    extract_part_features(part)

    assembly = load_step_assembly(args.assembly)
    extract_assembly_features(assembly)

    cfg = MatingConfig(
        exact_fit_tol_mm=args.exact_tol,
        max_clearance_mm=args.max_clearance,
    )
    matches = find_mating_instances(
        query_part=part,
        selected_face_ids=face_ids,
        assembly=assembly,
        config=cfg,
        auto_exclude_self=not args.include_self,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"{Path(args.part).stem}_in_{Path(args.assembly).stem}"

    report = {
        "query_part": part.name,
        "selected_face_ids": face_ids,
        "assembly": assembly.source_path,
        "num_assembly_parts": len(assembly.parts),
        "num_matches": len(matches),
        "matches": [m.to_dict() for m in matches],
    }
    json_path = out_dir / f"{stem}_matches.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Found {len(matches)} mating instance(s). Report saved to {json_path}")
    for m in matches:
        print(f"  [{m.match_id}] parts={m.part_names} local_faces={m.target_local_face_ids} -> {m.reasoning}")

    if not args.no_render:
        png_path = out_dir / f"{stem}_render.png"
        render_mating_results(part, face_ids, assembly, matches, png_path, title=stem)
        print(f"Render saved to {png_path}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "inspect":
        cmd_inspect(args.step)
        return 0
    if args.command == "match":
        return cmd_match(args)
    if args.command == "ui":
        from .ui import run_ui_server
        run_ui_server(host=args.host, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
