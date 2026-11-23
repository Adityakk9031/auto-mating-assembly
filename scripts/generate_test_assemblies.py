"""Generate reproducible multi-part STEP assemblies covering 6 mechanical mating scenarios."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.STEPCAFControl import STEPCAFControl_Writer

import build123d as bd

from automate.features import extract_part_features
from automate.models import SurfaceType
from automate.step_io import load_step_part


def _save_xcaf_assembly(parts: list[tuple[str, object]], out_path: Path) -> None:
    """Save named OCP shapes into a standard XCAF STEP assembly."""
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    root_label = shape_tool.NewShape()
    TDataStd_Name.Set_s(root_label, TCollection_ExtendedString(out_path.stem))

    for name, shape in parts:
        ocp_shape = shape.wrapped if hasattr(shape, "wrapped") else shape
        part_label = shape_tool.AddShape(ocp_shape, False)
        TDataStd_Name.Set_s(part_label, TCollection_ExtendedString(name))
        shape_tool.AddComponent(root_label, part_label, TopLoc_Location())

    shape_tool.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    writer.Write(str(out_path))


def _save_single_part(shape, out_path: Path) -> None:
    ocp_shape = shape.wrapped if hasattr(shape, "wrapped") else shape
    writer = STEPControl_Writer()
    writer.Transfer(ocp_shape, STEPControl_AsIs)
    writer.Write(str(out_path))


# ---------------------------------------------------------------------------
# Scenario 1: Lathe Chuck (3-jaw & 4-jaw) + Chuck Key (Exact Fit, Multi-Instance)
# ---------------------------------------------------------------------------
def build_chuck_and_key(out_dir: Path, num_pinions: int = 3) -> dict:
    sc_dir = out_dir / f"01_chuck_key_{num_pinions}slot"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Chuck body (outer R=62, bore R=18, height=40) with radial pinion holes at R_pin=14, z=20
    with bd.BuildPart() as chuck_bp:
        bd.Cylinder(radius=62, height=40, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        bd.Cylinder(radius=18, height=42, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        for idx in range(num_pinions):
            ang = idx * (360.0 / num_pinions)
            with bd.Locations(bd.Location((0, 0, 20)) * bd.Rotation(0, 0, ang) * bd.Location((48, 0, 0))):
                bd.Cylinder(radius=11.0, height=30, rotation=(0, 90, 0), mode=bd.Mode.SUBTRACT)
        # Decoy smaller radial hole (R=5)
        with bd.Locations(bd.Location((0, 0, 20)) * bd.Rotation(0, 0, 45) * bd.Location((48, 0, 0))):
            bd.Cylinder(radius=5.0, height=30, rotation=(0, 90, 0), mode=bd.Mode.SUBTRACT)

    # Pinions placed radially inside the chuck with 8.02mm x 8.02mm square key sockets
    parts: list[tuple[str, object]] = [("ChuckBody", chuck_bp.part)]
    for idx in range(num_pinions):
        ang = idx * (360.0 / num_pinions)
        with bd.BuildPart() as pin_bp:
            bd.Cylinder(radius=10.8, height=22, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
            with bd.Locations((0, 0, 10)):
                bd.Box(8.02, 8.02, 13, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        loc = bd.Location((0, 0, 20)) * bd.Rotation(0, 0, ang) * bd.Location((40, 0, 0)) * bd.Rotation(0, 90, 0)
        parts.append((f"ChuckPinion_{idx+1}", pin_bp.part.moved(loc)))

    # Decoy pinion with wrong square socket size (11.5mm instead of 8.0mm)
    with bd.BuildPart() as decoy_pin:
        bd.Cylinder(radius=10.8, height=22, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        with bd.Locations((0, 0, 10)):
            bd.Box(11.5, 11.5, 13, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
    parts.append(("DecoyPinion_WrongSize", decoy_pin.part.moved(bd.Location((95, 0, 0)))))

    # T-handle Chuck Key at an arbitrary SE(3) pose above/beside the chuck
    with bd.BuildPart() as key_bp:
        # Square drive tip: 8.0mm x 8.0mm x 16mm (z=0..16)
        bd.Box(8.0, 8.0, 16.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        # Cylindrical shank: R=5.5, z=16..70
        with bd.Locations((0, 0, 16.0)):
            bd.Cylinder(radius=5.5, height=54.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        # T-bar handle at z=64
        with bd.Locations((0, 0, 64.0)):
            bd.Cylinder(radius=4.5, height=76.0, rotation=(90, 0, 0))

    key_pose = bd.Location((75, -55, 45)) * bd.Rotation(35, -25, 50)
    key_moved = key_bp.part.moved(key_pose)
    parts.append(("ChuckKey", key_moved))

    part_path = sc_dir / "chuck_key.stp"
    asm_path = sc_dir / "chuck_assembly.stp"
    _save_single_part(key_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    # Identify the 4 square tip side faces automatically from the local part geometry
    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    # In key_bp local frame before move, the 4 tip faces have area = 8*16 = 128 mm^2 and are PLANEs
    tip_face_ids = [
        f.local_face_id for f in p_data.faces
        if f.surface_type == SurfaceType.PLANE and 120.0 <= f.area <= 136.0
    ]
    return {
        "name": f"chuck_key_{num_pinions}slot",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": tip_face_ids,
        "expected_matches": num_pinions,
        "expected_fit": "EXACT_FIT",
    }


# ---------------------------------------------------------------------------
# Scenario 2: Bench Vice Sliding Jaw + Vice Body (Sliding/Clearance Fit)
# ---------------------------------------------------------------------------
def build_bench_vice(out_dir: Path) -> dict:
    sc_dir = out_dir / "02_bench_vice"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Vice body: base 180 x 80 x 35 with a central guide channel 24.5mm wide x 14mm deep (0.25mm/side clearance!)
    # Plus fixed jaw at x=-75..-50, plus a decoy wider mounting pocket 34mm wide on the side
    with bd.BuildPart() as body_bp:
        bd.Box(180, 80, 35, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        # Fixed jaw block on left end
        with bd.Locations((-75, 0, 35)):
            bd.Box(30, 80, 38, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        # Guide channel along X (length 150, width 24.5, depth 14 from z=21..35)
        with bd.Locations((15, 0, 21)):
            bd.Box(150, 24.5, 15, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        # Decoy wider cross-slot (width 34mm) underneath
        with bd.Locations((0, 0, 0)):
            bd.Box(34, 82, 8, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)

    # Sliding jaw: jaw block + 24.0mm wide x 48mm long guide tongue underneath (0.50mm total lateral clearance = 0.25mm/side)
    # Tongue height 14.3mm inside 14.0mm deep channel gives 0.30mm deck clearance above z=35.0
    with bd.BuildPart() as jaw_bp:
        bd.Box(28, 80, 38, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        with bd.Locations((0, 0, -14.3)):
            bd.Box(48, 24.0, 14.3, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))

    # Place jaw rotated/translated away from vice body to verify SE(3) invariance
    jaw_moved = jaw_bp.part.moved(bd.Location((110, 90, 25)) * bd.Rotation(0, 0, 42))

    # Leadscrew decoy cylinder
    with bd.BuildPart() as screw_bp:
        bd.Cylinder(radius=7.0, height=160, rotation=(0, 90, 0))

    parts = [
        ("ViceBody", body_bp.part),
        ("LeadScrew", screw_bp.part.moved(bd.Location((10, 0, 15)))),
        ("SlidingJaw", jaw_moved),
    ]

    part_path = sc_dir / "sliding_jaw.stp"
    asm_path = sc_dir / "bench_vice_assembly.stp"
    _save_single_part(jaw_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    # Select the 3 sliding faces of the guide tongue: bottom face (area = 48*24 = 1152) and two side cheeks (area = 48*14.3 = 686.4)
    slide_face_ids = [
        f.local_face_id for f in p_data.faces
        if f.surface_type == SurfaceType.PLANE and (abs(f.area - 1152.0) < 10.0 or abs(f.area - 686.4) < 10.0)
    ]
    return {
        "name": "bench_vice_sliding_jaw",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": slide_face_ids,
        "expected_matches": 1,
        "expected_fit": "SLIDING_CLEARANCE_FIT",
    }


# ---------------------------------------------------------------------------
# Scenario 3: Keyed Shaft + Matching Keyed Hubs (Mixed Cylinder + Plane Keyway)
# ---------------------------------------------------------------------------
def build_keyed_shaft(out_dir: Path) -> dict:
    sc_dir = out_dir / "03_keyed_shaft"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Motor shaft R=12mm, length=70mm, with an integral protruding parallel feather key (6mm wide x 4mm proud x 36mm long)
    with bd.BuildPart() as shaft_bp:
        bd.Cylinder(radius=12.0, height=70.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        with bd.Locations((12.0, 0, 18.0)):
            bd.Box(6.0, 6.0, 36.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))

    # Matching Hub 1 & Hub 2: bore R=12.02mm + matching keyway slot (width 6.03mm, outer radius 15.02mm matching key top at x=15.0)
    def make_keyed_hub(bore_r: float, key_w: float, outer_r: float) -> object:
        with bd.BuildPart() as hub_bp:
            bd.Cylinder(radius=outer_r, height=28.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
            bd.Cylinder(radius=bore_r, height=30.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
            with bd.Locations((bore_r, 0, -1)):
                bd.Box(6.0, key_w, 30.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        return hub_bp.part

    hub1 = make_keyed_hub(12.02, 6.03, 30.0).moved(bd.Location((-60, 40, 10)) * bd.Rotation(90, 0, 30))
    hub2 = make_keyed_hub(12.02, 6.03, 34.0).moved(bd.Location((65, -30, 20)) * bd.Rotation(0, 90, -45))
    decoy_plain = bd.Cylinder(radius=28, height=28) - bd.Cylinder(radius=12.02, height=30)  # no keyway!
    decoy_wrong_size = make_keyed_hub(16.0, 8.0, 34.0).moved(bd.Location((0, -75, 0)))

    shaft_moved = shaft_bp.part.moved(bd.Location((0, 70, 0)) * bd.Rotation(20, 45, 0))
    parts = [
        ("KeyedHub_A", hub1),
        ("KeyedHub_B", hub2),
        ("PlainCollar_NoKeyway", decoy_plain.moved(bd.Location((-55, -60, 0)))),
        ("WrongSizeHub_32mm", decoy_wrong_size),
        ("KeyedMotorShaft", shaft_moved),
    ]

    part_path = sc_dir / "keyed_shaft.stp"
    asm_path = sc_dir / "keyed_shaft_assembly.stp"
    _save_single_part(shaft_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    # Select the shaft cylinder (convex, R=12) + the two parallel key flank faces (area = 36*6? Wait: box overlaps shaft slightly, outer width ~3*36=108, side faces ~180)
    # Let's select shaft cylinder + 3 outer key faces (left, right, top)
    sel_ids = []
    for f in p_data.faces:
        if f.surface_type == SurfaceType.CYLINDER and f.sense == 1 and abs(f.radius - 12.0) < 0.2:
            sel_ids.append(f.local_face_id)
        elif f.surface_type == SurfaceType.PLANE and f.area > 50.0 and f.area < 250.0:
            # Exclude circular end caps (area = pi*12^2 = 452) and key ends (area ~ 20)
            sel_ids.append(f.local_face_id)

    return {
        "name": "keyed_shaft_to_hubs",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": sel_ids,
        "expected_matches": 2,
        "expected_fit": "EXACT_FIT",
    }


# ---------------------------------------------------------------------------
# Scenario 4: Pulley-Wheel upon Motor Shaft (Cylindrical Bore + Shoulder)
# ---------------------------------------------------------------------------
def build_pulley_on_motor_shaft(out_dir: Path) -> dict:
    sc_dir = out_dir / "04_pulley_motor_shaft"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Pulley wheel: outer R=42, hub R=20, bore R=10.0mm (20mm diameter) with axial shoulder
    with bd.BuildPart() as pulley_bp:
        bd.Cylinder(radius=42.0, height=18.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        with bd.Locations((0, 0, 18.0)):
            bd.Cylinder(radius=20.0, height=12.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        # Stepped bore: R=10.0 from z=0..22, R=7.0 from z=22..30 (creates shoulder ring at z=22)
        bd.Cylinder(radius=10.0, height=22.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)
        with bd.Locations((0, 0, 22.0)):
            bd.Cylinder(radius=7.0, height=9.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN), mode=bd.Mode.SUBTRACT)

    # Motor 1 (matching shaft R=9.92mm -> 0.08mm radial clearance / 0.16mm diametral clearance = SLIDING_CLEARANCE_FIT, plus shoulder!)
    def make_motor(shaft_r: float) -> object:
        with bd.BuildPart() as m_bp:
            bd.Cylinder(radius=36.0, height=55.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
            # Shaft step R=15 then shaft R=shaft_r
            with bd.Locations((0, 0, 55.0)):
                bd.Cylinder(radius=15.0, height=12.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
            with bd.Locations((0, 0, 67.0)):
                bd.Cylinder(radius=shaft_r, height=26.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        return m_bp.part

    motor_match = make_motor(9.85).moved(bd.Location((-60, 0, 0)) * bd.Rotation(0, 90, 0))
    motor_wrong = make_motor(13.5).moved(bd.Location((60, 50, 0)) * bd.Rotation(90, 0, 0))
    pulley_moved = pulley_bp.part.moved(bd.Location((40, -60, 30)) * bd.Rotation(40, 20, 10))

    parts = [
        ("ElectricMotor_20mmShaft", motor_match),
        ("HeavyMotor_27mmShaft", motor_wrong),
        ("DrivePulley", pulley_moved),
    ]

    part_path = sc_dir / "pulley_wheel.stp"
    asm_path = sc_dir / "motor_pulley_assembly.stp"
    _save_single_part(pulley_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    # Select R=10 bore cylinder + shoulder annular plane (area = pi*(10^2 - 7^2) = 160.2)
    sel_ids = []
    for f in p_data.faces:
        if f.surface_type == SurfaceType.CYLINDER and f.sense == -1 and abs(f.radius - 10.0) < 0.2:
            sel_ids.append(f.local_face_id)
        elif f.surface_type == SurfaceType.PLANE and abs(f.area - math.pi * (100.0 - 49.0)) < 15.0:
            sel_ids.append(f.local_face_id)

    return {
        "name": "pulley_on_motor_shaft",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": sel_ids,
        "expected_matches": 1,
        "expected_fit": "SLIDING_CLEARANCE_FIT",
    }


# ---------------------------------------------------------------------------
# Scenario 5: V-Belt Inner Faces upon V-Pulley Groove (Conical + Cylindrical)
# ---------------------------------------------------------------------------
def build_pulley_and_vbelt(out_dir: Path) -> dict:
    sc_dir = out_dir / "05_pulley_vbelt"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # V-Pulley revolved profile: outer R=40, groove bottom R=31, width=18, 38-deg V-groove (19 deg half-angle)
    def make_vpulley(r_bot: float, r_top: float, half_ang_deg: float) -> object:
        # Groove half-width at bottom = 3.0mm
        w_bot = 3.0
        w_top = w_bot + (r_top - r_bot) * math.tan(math.radians(half_ang_deg))
        half_thick = w_top + 3.5
        pts = [
            (10.0, -half_thick),
            (r_top, -half_thick),
            (r_top, -w_top),
            (r_bot, -w_bot),
            (r_bot, w_bot),
            (r_top, w_top),
            (r_top, half_thick),
            (10.0, half_thick),
        ]
        with bd.BuildPart() as vp:
            with bd.BuildSketch(bd.Plane.XZ):
                with bd.BuildLine():
                    bd.Polyline(*pts, close=True)
                bd.make_face()
            bd.revolve(axis=bd.Axis.Z, revolution_arc=360.0)
        return vp.part

    pulley_match = make_vpulley(31.0, 40.0, 19.0)
    pulley_small = make_vpulley(19.0, 26.0, 19.0).moved(bd.Location((95, 0, 0)))
    pulley_wrong_angle = make_vpulley(31.0, 40.0, 32.0).moved(bd.Location((-95, 0, 0)))

    # V-belt wrap segment (180-deg arc matching r_bot=31.05..r_top=39.5, half_angle=19 deg)
    w_bot = 3.0
    r_b = 31.05
    r_t = 39.5
    w_t = w_bot + (r_t - r_b) * math.tan(math.radians(19.0))
    belt_pts = [
        (r_b, -w_bot),
        (r_t, -w_t),
        (r_t, w_t),
        (r_b, w_bot),
    ]
    with bd.BuildPart() as belt_bp:
        with bd.BuildSketch(bd.Plane.XZ):
            with bd.BuildLine():
                bd.Polyline(*belt_pts, close=True)
            bd.make_face()
        bd.revolve(axis=bd.Axis.Z, revolution_arc=180.0)

    belt_moved = belt_bp.part.moved(bd.Location((0, 75, 25)) * bd.Rotation(25, 0, -40))

    parts = [
        ("VPulley_R40_38deg", pulley_match),
        ("VPulley_Small_R26", pulley_small),
        ("VPulley_WrongAngle_64deg", pulley_wrong_angle),
        ("VBelt_WrapArc", belt_moved),
    ]

    part_path = sc_dir / "vbelt_arc.stp"
    asm_path = sc_dir / "vbelt_pulley_assembly.stp"
    _save_single_part(belt_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    # Select the 2 conical belt flanks (CONE) + inner cylinder (CYLINDER, r~31.05)
    sel_ids = [
        f.local_face_id for f in p_data.faces
        if f.surface_type == SurfaceType.CONE
        or (f.surface_type == SurfaceType.CYLINDER and f.sense == -1 and abs(f.radius - r_b) < 0.2)
    ]
    return {
        "name": "vbelt_on_vpulley",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": sel_ids,
        "expected_matches": 1,
        "expected_fit": "EXACT_FIT",
    }


# ---------------------------------------------------------------------------
# Scenario 6: Spherical Ball Joint + Freeform Spline Surface
# ---------------------------------------------------------------------------
def build_sphere_and_freeform(out_dir: Path) -> dict:
    sc_dir = out_dir / "06_sphere_freeform"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Ball joint: sphere R=14mm on a cylindrical stem
    with bd.BuildPart() as ball_bp:
        bd.Sphere(radius=14.0)
        with bd.Locations((0, 0, 12)):
            bd.Cylinder(radius=5.0, height=24.0, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))

    # Two matching socket cups (R=14.02mm) + one wrong-size socket (R=18mm)
    def make_socket(r_sock: float) -> object:
        with bd.BuildPart() as sock_bp:
            bd.Box(40, 40, 22, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
            with bd.Locations((0, 0, 22)):
                bd.Sphere(radius=r_sock, mode=bd.Mode.SUBTRACT)
        return sock_bp.part

    sock1 = make_socket(14.02)
    sock2 = make_socket(14.02).moved(bd.Location((60, 0, 0)) * bd.Rotation(0, 45, 0))
    sock_decoy = make_socket(18.0).moved(bd.Location((-60, 0, 0)))

    ball_moved = ball_bp.part.moved(bd.Location((0, 65, 20)) * bd.Rotation(55, 20, 0))

    parts = [
        ("BallSocket_1", sock1),
        ("BallSocket_2", sock2),
        ("DecoySocket_R18", sock_decoy),
        ("BallStud", ball_moved),
    ]
    part_path = sc_dir / "ball_stud.stp"
    asm_path = sc_dir / "ball_joint_assembly.stp"
    _save_single_part(ball_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    sphere_ids = [f.local_face_id for f in p_data.faces if f.surface_type == SurfaceType.SPHERE]

    return {
        "name": "spherical_ball_joint",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": sphere_ids,
        "expected_matches": 2,
        "expected_fit": "EXACT_FIT",
    }


# ---------------------------------------------------------------------------
# Scenario 7: Freeform BSpline Cam/Loft Boss into Freeform Pocket
# ---------------------------------------------------------------------------
def build_freeform_mate(out_dir: Path) -> dict:
    sc_dir = out_dir / "07_freeform_spline"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Freeform lofted cam boss (asymmetric kidney/cam spline profile lofted with twist/taper)
    def make_cam_loft(scale: float = 1.0) -> object:
        pts_bot = [
            (10 * scale, 0),
            (7 * scale, 9 * scale),
            (-4 * scale, 11 * scale),
            (-12 * scale, 2 * scale),
            (-6 * scale, -8 * scale),
            (6 * scale, -7 * scale),
        ]
        pts_top = [
            (8 * scale, 1 * scale),
            (5 * scale, 7.5 * scale),
            (-3 * scale, 9 * scale),
            (-9 * scale, 1.5 * scale),
            (-5 * scale, -6.5 * scale),
            (5 * scale, -5.5 * scale),
        ]
        with bd.BuildPart() as lp:
            with bd.BuildSketch(bd.Plane.XY):
                with bd.BuildLine():
                    bd.Spline(*pts_bot, periodic=True)
                bd.make_face()
            with bd.BuildSketch(bd.Plane.XY.offset(12.0)):
                with bd.BuildLine():
                    bd.Spline(*pts_top, periodic=True)
                bd.make_face()
            bd.loft()
        return lp.part

    cam_boss = make_cam_loft(1.0)
    with bd.BuildPart() as plug_bp:
        bd.Box(32, 32, 8, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MAX))
        bd.add(cam_boss)

    with bd.BuildPart() as sock_bp:
        bd.Box(36, 36, 18, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        bd.add(make_cam_loft(1.002), mode=bd.Mode.SUBTRACT)

    with bd.BuildPart() as decoy_sock_bp:
        bd.Box(44, 44, 18, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        bd.add(make_cam_loft(1.35), mode=bd.Mode.SUBTRACT)

    plug_moved = plug_bp.part.moved(bd.Location((55, 45, 20)) * bd.Rotation(20, -30, 50))
    sock_moved = sock_bp.part.moved(bd.Location((-25, -10, 0)) * bd.Rotation(0, 0, 35))
    decoy_moved = decoy_sock_bp.part.moved(bd.Location((65, -40, 0)))

    parts = [
        ("FreeformSocket_Match", sock_moved),
        ("FreeformSocket_DecoyLarge", decoy_moved),
        ("FreeformPlug", plug_moved),
    ]
    part_path = sc_dir / "freeform_plug.stp"
    asm_path = sc_dir / "freeform_assembly.stp"
    _save_single_part(plug_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    ff_ids = [f.local_face_id for f in p_data.faces if f.surface_type == SurfaceType.FREEFORM]
    return {
        "name": "freeform_bspline_mate",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": ff_ids,
        "expected_matches": 1,
        "expected_fit": "EXACT_FIT",
    }


# ---------------------------------------------------------------------------
# Scenario 8: 60-Deg Dovetail Machine-Tool Cross-Slide (Non-Orthogonal Planes)
# ---------------------------------------------------------------------------
def build_dovetail_slide(out_dir: Path) -> dict:
    sc_dir = out_dir / "08_dovetail_slide"
    sc_dir.mkdir(parents=True, exist_ok=True)

    # Dovetail geometry: bottom width > top neck width (60-deg undercut angle)
    # Channel depth h=12mm, cot(60deg)=0.57735 -> flare per side = 12 * 0.57735 = 6.928mm
    def make_dovetail_prism(w_neck: float, h: float, flare: float, length: float) -> object:
        w_bot = w_neck + 2.0 * flare
        pts = [
            (-w_bot / 2.0, 0.0),
            (w_bot / 2.0, 0.0),
            (w_neck / 2.0, h),
            (-w_neck / 2.0, h),
        ]
        with bd.BuildPart() as dp:
            with bd.BuildSketch(bd.Plane.YZ):
                with bd.BuildLine():
                    bd.Polyline(*pts, close=True)
                bd.make_face()
            bd.extrude(amount=length / 2.0, both=True)
        return dp.part

    flare_60 = 12.0 / math.tan(math.radians(60.0))
    cut_60 = make_dovetail_prism(27.0, 12.1, flare_60, 164.0).moved(bd.Location((0, 0, 16.0)))
    # Female dovetail ways: neck=27.0mm, depth=12.0mm (~0.20mm perpendicular cheek clearance)
    with bd.BuildPart() as base_bp:
        bd.Box(160, 76, 28, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        bd.add(cut_60, mode=bd.Mode.SUBTRACT)

    # Decoy dovetail ways with wrong angle (45-deg flare instead of 60-deg)
    flare_45 = 12.0 / math.tan(math.radians(45.0))
    cut_45 = make_dovetail_prism(27.0, 12.1, flare_45, 164.0).moved(bd.Location((0, 0, 16.0)))
    with bd.BuildPart() as decoy_bp:
        bd.Box(160, 76, 28, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))
        bd.add(cut_45, mode=bd.Mode.SUBTRACT)

    # Male dovetail carriage: neck=26.0mm (0.4mm total horizontal clearance -> 0.173mm perpendicular cheek gap)
    # Height=12.3mm so upper carriage block rides 0.3mm above z=28.0 deck
    flare_60_carriage = 12.3 / math.tan(math.radians(60.0))
    tongue_60 = make_dovetail_prism(26.0, 12.3, flare_60_carriage, 54.0)
    with bd.BuildPart() as carriage_bp:
        bd.add(tongue_60)
        with bd.Locations((0, 0, 12.3)):
            bd.Box(54, 68, 18, align=(bd.Align.CENTER, bd.Align.CENTER, bd.Align.MIN))

    carriage_moved = carriage_bp.part.moved(bd.Location((0, 95, 35)) * bd.Rotation(15, 25, -50))

    parts = [
        ("DovetailWays_60deg", base_bp.part),
        ("DecoyWays_45deg", decoy_bp.part.moved(bd.Location((0, -95, 0)))),
        ("DovetailCarriage", carriage_moved),
    ]
    part_path = sc_dir / "dovetail_carriage.stp"
    asm_path = sc_dir / "dovetail_assembly.stp"
    _save_single_part(carriage_moved, part_path)
    _save_xcaf_assembly(parts, asm_path)

    p_data = load_step_part(part_path)
    extract_part_features(p_data)
    # Select the 2 slanted 60-deg dovetail cheek faces (area = 54 * 12.3 / sin(60deg) = 767.0) + bottom face (area = 54 * (26 + 2*7.101) = 2170.9)
    cheek_area = 54.0 * (12.3 / math.sin(math.radians(60.0)))
    bot_area = 54.0 * (26.0 + 2.0 * flare_60_carriage)
    sel_ids = [
        f.local_face_id for f in p_data.faces
        if f.surface_type == SurfaceType.PLANE and (abs(f.area - cheek_area) < 15.0 or abs(f.area - bot_area) < 15.0)
    ]
    return {
        "name": "dovetail_cross_slide",
        "part": str(part_path),
        "assembly": str(asm_path),
        "faces": sel_ids,
        "expected_matches": 1,
        "expected_fit": "SLIDING_CLEARANCE_FIT",
    }


def main() -> None:
    out_dir = Path("scenarios")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = [
        build_chuck_and_key(out_dir, num_pinions=3),
        build_chuck_and_key(out_dir, num_pinions=4),
        build_bench_vice(out_dir),
        build_keyed_shaft(out_dir),
        build_pulley_on_motor_shaft(out_dir),
        build_pulley_and_vbelt(out_dir),
        build_sphere_and_freeform(out_dir),
        build_freeform_mate(out_dir),
        build_dovetail_slide(out_dir),
    ]
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"Generated {len(manifests)} test scenarios -> {manifest_path}")
    for m in manifests:
        print(f"  - {m['name']}: faces={m['faces']}, expected={m['expected_matches']} ({m['expected_fit']})")


if __name__ == "__main__":
    main()
