from __future__ import annotations

import json
from pathlib import Path
import pytest

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
import build123d as bd

from automate.features import extract_assembly_features, extract_part_features
from automate.matcher import find_mating_instances
from automate.models import FitType, MatingConfig, PartData
from automate.step_io import load_step_assembly, load_step_part


SCENARIOS = json.loads(Path("scenarios/manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("item", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_scenario_matching_and_fit_classification(item: dict) -> None:
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

    assert len(matches) == item["expected_matches"], (
        f"{item['name']}: expected {item['expected_matches']} matches, got {len(matches)}: "
        f"{[(m.part_names, m.reasoning) for m in matches]}"
    )

    for m in matches:
        assert m.fit_type.value == item["expected_fit"], (
            f"{item['name']}: expected {item['expected_fit']}, got {m.fit_type.value} ({m.reasoning})"
        )
        assert m.overlap_score >= 0.70
        assert m.normal_error_deg <= 2.5
        for pname in m.part_names:
            assert "Decoy" not in pname and "Wrong" not in pname and "Plain" not in pname


def test_se3_pose_invariance_on_chuck_key() -> None:
    item = next(s for s in SCENARIOS if s["name"] == "chuck_key_3slot")
    part = load_step_part(item["part"])

    # Apply an arbitrary large rigid transformation in SE(3)
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(10, -20, 5), gp_Dir(0.5, 0.7, -0.5)), 2.13)
    trsf2 = gp_Trsf()
    trsf2.SetTranslation(gp_Vec(-450.0, 320.0, -180.0))
    trsf.Multiply(trsf2)

    moved_shape = BRepBuilderAPI_Transform(part.shape, trsf, True).Shape()
    moved_part = PartData(part_id=-1, name="MovedKey", shape=moved_shape)
    extract_part_features(moved_part)

    asm = load_step_assembly(item["assembly"])
    extract_assembly_features(asm)

    matches = find_mating_instances(moved_part, item["faces"], asm, MatingConfig())
    assert len(matches) == 3
    matched_parts = sorted(p for m in matches for p in m.part_names)
    assert matched_parts == ["ChuckPinion_1", "ChuckPinion_2", "ChuckPinion_3"]

    # Verify chuck key handle points radially OUTWARD (never buried inside the chuck)
    p_com = sum(f.centroid for f in moved_part.faces) / len(moved_part.faces)
    tip_com = sum(moved_part.faces[i].centroid for i in item["faces"]) / len(item["faces"])
    for m in matches:
        R = m.transform_4x4[:3, :3]
        t = m.transform_4x4[:3, 3]
        w_tip = R @ tip_com + t
        w_com = R @ p_com + t
        assert (w_com[0] ** 2 + w_com[1] ** 2) > (w_tip[0] ** 2 + w_tip[1] ** 2)


def test_freeform_rigid_0dof_and_se3_invariance() -> None:
    item = next(s for s in SCENARIOS if s["name"] == "freeform_bspline_mate")
    part = load_step_part(item["part"])

    # Rotate freeform plug by an arbitrary 3D axis-angle
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(5, -10, 15), gp_Dir(-0.4, 0.6, 0.69)), 1.75)
    moved_shape = BRepBuilderAPI_Transform(part.shape, trsf, True).Shape()
    moved_part = PartData(part_id=-1, name="MovedFreeformPlug", shape=moved_shape)
    extract_part_features(moved_part)

    asm = load_step_assembly(item["assembly"])
    extract_assembly_features(asm)

    matches = find_mating_instances(moved_part, item["faces"], asm, MatingConfig())
    assert len(matches) == 1
    assert matches[0].part_names == ["FreeformSocket_Match"]
    assert matches[0].kinematic_pair.value == "RIGID_0DOF"
    assert matches[0].dof_translation == 0
    assert matches[0].dof_rotation == 0


def test_seam_split_half_cylinder_bore_merging() -> None:
    """Verify a target bore split into two 180-deg half-cylinder faces merges into 1 unified match."""
    import sys
    import tempfile
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.generate_test_assemblies import _save_single_part, _save_xcaf_assembly

    # Build a single solid housing whose R=10mm bore consists of 2 seam-split 180-deg half-cylinder faces
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    with bd.BuildPart() as half1_bp:
        bd.Box(50, 25, 24, align=(bd.Align.CENTER, bd.Align.MIN, bd.Align.MIN))
        bd.Cylinder(radius=10.0, height=26.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
    with bd.BuildPart() as half2_bp:
        bd.Box(50, 25, 24, align=(bd.Align.CENTER, bd.Align.MAX, bd.Align.MIN))
        bd.Cylinder(radius=10.0, height=26.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)

    sew = BRepBuilderAPI_Sewing(1e-3)
    for f in half1_bp.part.faces():
        if abs(f.center().Y) > 1e-3:
            sew.Add(f.wrapped)
    for f in half2_bp.part.faces():
        if abs(f.center().Y) > 1e-3:
            sew.Add(f.wrapped)
    sew.Perform()
    sewn_shape = sew.SewedShape()
    exp = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
    housing_solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell(exp.Current())).Solid()

    with bd.BuildPart() as pin_bp:
        bd.Cylinder(radius=9.98, height=24.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))

    pin_moved = pin_bp.part.moved(bd.Location((70, -40, 20)) * bd.Rotation(35, 25, 0))
    parts = [("SplitSeamHousing", housing_solid), ("CylPin", pin_moved)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        p_path = tmp_p / "pin.stp"
        a_path = tmp_p / "split_bore_asm.stp"
        _save_single_part(pin_moved, p_path)
        _save_xcaf_assembly(parts, a_path)

        q_part = load_step_part(p_path)
        extract_part_features(q_part)
        cyl_ids = [f.local_face_id for f in q_part.faces if f.surface_type.value == "CYLINDER"]
        assert len(cyl_ids) == 1

        asm = load_step_assembly(a_path)
        extract_assembly_features(asm)
        # Verify the housing actually has 2 concave half-cylinder faces
        housing_cyls = [f for f in asm.parts[0].faces if f.surface_type.value == "CYLINDER" and f.sense == -1]
        assert len(housing_cyls) == 2

        matches = find_mating_instances(q_part, cyl_ids, asm, MatingConfig())
        assert len(matches) == 1
        assert len(matches[0].target_local_face_ids) == 2


def test_negative_oversized_key_rejects_smaller_sockets() -> None:
    # Build a 10.5mm square key (too wide for 8.02mm sockets, too narrow for 11.5mm decoy)
    item = next(s for s in SCENARIOS if s["name"] == "chuck_key_3slot")
    asm = load_step_assembly(item["assembly"])
    extract_assembly_features(asm)

    with bd.BuildPart() as bad_key_bp:
        bd.Box(10.5, 10.5, 16.0)

    bad_part = PartData(part_id=-1, name="OversizedKey", shape=bad_key_bp.part.wrapped)
    extract_part_features(bad_part)
    side_ids = [
        f.local_face_id for f in bad_part.faces if abs(f.area - 10.5 * 16.0) < 1.0
    ]
    assert len(side_ids) == 4

    matches = find_mating_instances(bad_part, side_ids, asm, MatingConfig())
    assert len(matches) == 0


def test_cross_part_split_guide_channel() -> None:
    # Two separate bolted guide rails + base plate forming a 20.4mm sliding channel
    import sys
    import tempfile
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.generate_test_assemblies import _save_single_part, _save_xcaf_assembly

    with bd.BuildPart() as rail_l:
        bd.Box(140, 15, 15, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    with bd.BuildPart() as rail_r:
        bd.Box(140, 15, 15, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
    with bd.BuildPart() as base:
        bd.Box(140, 60, 12, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX))
    with bd.BuildPart() as slider:
        bd.Box(40, 20.0, 14.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))

    # Left rail y in [-25.2, -10.2], Right rail y in [10.2, 25.2] -> channel width = 20.4mm
    parts = [
        ("BasePlate", base.part),
        ("LeftGuideRail", rail_l.part.moved(bd.Location((0, -17.7, 0)))),
        ("RightGuideRail", rail_r.part.moved(bd.Location((0, 17.7, 0)))),
        ("SliderBlock", slider.part.moved(bd.Location((100, 100, 50)) * bd.Rotation(0, 0, 35))),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        part_path = tmp_p / "slider.stp"
        asm_path = tmp_p / "split_channel.stp"
        _save_single_part(parts[-1][1], part_path)
        _save_xcaf_assembly(parts, asm_path)

        q_part = load_step_part(part_path)
        extract_part_features(q_part)
        # Select bottom (area=800) + two side cheeks (area=560)
        sel_ids = [
            f.local_face_id for f in q_part.faces
            if abs(f.area - 800.0) < 5.0 or abs(f.area - 560.0) < 5.0
        ]
        # Keep only 1 bottom + 2 sides (exclude top face which also has area 800)
        bottom_faces = [f for f in q_part.faces if abs(f.area - 800.0) < 5.0]
        side_faces = [f for f in q_part.faces if abs(f.area - 560.0) < 5.0]
        # Pick the bottom face whose normal is opposite to the top
        sel_ids = [bottom_faces[0].local_face_id] + [f.local_face_id for f in side_faces]

        asm = load_step_assembly(asm_path)
        extract_assembly_features(asm)
        matches = find_mating_instances(q_part, sel_ids, asm, MatingConfig())

        assert len(matches) >= 1
        cross_matches = [m for m in matches if len(m.part_names) == 3]
        assert len(cross_matches) == 1
        assert cross_matches[0].fit_type == FitType.SLIDING_CLEARANCE_FIT
        assert set(cross_matches[0].part_names) == {"BasePlate", "LeftGuideRail", "RightGuideRail"}
