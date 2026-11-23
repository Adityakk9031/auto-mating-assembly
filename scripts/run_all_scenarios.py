"""Run auto-mating on all generated scenarios and save JSON + 3D PNG renders to outputs/."""

from __future__ import annotations

import json
import time
from pathlib import Path

from automate.features import extract_assembly_features, extract_part_features
from automate.matcher import find_mating_instances
from automate.models import MatingConfig
from automate.step_io import load_step_assembly, load_step_part
from automate.visualizer import render_mating_results


def main() -> None:
    manifest = json.loads(Path("scenarios/manifest.json").read_text(encoding="utf-8"))
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for item in manifest:
        name = item["name"]
        t0 = time.perf_counter()
        part = load_step_part(item["part"])
        extract_part_features(part)
        asm = load_step_assembly(item["assembly"])
        extract_assembly_features(asm)

        matches = find_mating_instances(
            query_part=part,
            selected_face_ids=item["faces"],
            assembly=asm,
            config=MatingConfig(),
        )
        elapsed = time.perf_counter() - t0

        report = {
            "scenario": name,
            "query_part": part.name,
            "selected_face_ids": item["faces"],
            "assembly_parts": [p.name for p in asm.parts],
            "expected_matches": item["expected_matches"],
            "expected_fit": item["expected_fit"],
            "num_matches": len(matches),
            "elapsed_sec": round(elapsed, 3),
            "matches": [m.to_dict() for m in matches],
        }
        (out_dir / f"{name}_matches.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        png_path = out_dir / f"{name}_render.png"
        render_mating_results(part, item["faces"], asm, matches, png_path, title=name)

        fits = [m.fit_type.value for m in matches]
        parts_hit = [m.part_names for m in matches]
        print(
            f"[{name}] {len(matches)}/{item['expected_matches']} matches in {elapsed:.2f}s | "
            f"fits={fits} | parts={parts_hit} -> {png_path}"
        )
        summary.append(report)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
