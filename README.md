# Auto-Mating in STEP Assemblies (`automate-cad`)

SE(3)-invariant B-Rep assembly mating engine built on Open CASCADE Technology (`cadquery-ocp` / OCCT 8.0), `scipy`, and `vtk`. Given a single-part STEP file, a set of selected face IDs on that part, and an assembly STEP file, `automate-cad` finds **every** compatible mating location in the assembly, identifies the matching part(s) and face IDs, solves the aligning rigid transform $T \in SE(3)$, distinguishes **exact/interference fits** from **sliding/clearance fits** and **kinematic pair DOFs**, and renders highlighted 3D PNG reports.

## Features

- **Any-face-group query**: Works for arbitrary face selections across `PLANE`, `CYLINDER`, `CONE`, `SPHERE`, `TORUS`, and `FREEFORM` (`BSpline` / `Bezier` / extruded / revolved) surfaces.
- **$SE(3)$ Pose Invariance**: Matches using intrinsic surface parameters and pairwise relative invariants (normal/axis angles, signed normal-plane gaps, half-space chirality, axis-to-axis line distances, cone apex separation), independent of world position or orientation.
- **Exact vs. Sliding/Clearance Fit Classification**: Distinguishes coinciding/interference contacts (`EXACT_FIT`, `INTERFERENCE_FIT`) from controlled-gap sliding pairs (`SLIDING_CLEARANCE_FIT`) using invariant opposing-surface clearance, radial clearance, and constraint Jacobian nullspace DOFs (`RIGID_0DOF`, `PRISMATIC_SLIDE`, `REVOLUTE`, `CYLINDRICAL`, `PLANAR`, `SPHERICAL`).
- **Multi-Instance, Seam-Split & Cross-Part Discovery**: Reports all valid matching instances (e.g., all 3 or 4 chuck pinion sockets, multiple keyed hubs, or split guide channels spanning bolted parts), merges seam-split co-surface B-Rep faces (such as two $180^\circ$ half-cylinder bore faces) into unified logical matches, and deduplicates symmetric face permutations on the same pocket.
- **Full-Part Solid Collision & 1-DOF Seating**: Rejects upside-down or inward-penetrating orientations by probing interior material points on unselected body faces against target solids (`BRepClass3d_SolidClassifier`) and automatically seats longer keys above blind pocket floors along 1-DOF sliding axes.
- **3-Panel 3D Visual Verification**: Offscreen VTK + B-Rep renderer produces 3-panel PNG views showing (1) query part & selected faces in green, (2) assembly target matches in green (`EXACT_FIT`), cyan (`SLIDING_CLEARANCE_FIT`), or amber (`INTERFERENCE_FIT`), and (3) **$SE(3)$ Mated Pose Verification** with the query part placed into each matched assembly location in gold.

---

## Installation

Using `uv` (Python 3.11+):

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e .
```

---

## Quick Start

### 1. Launch the Interactive 3D CAD Web UI

```powershell
.venv\Scripts\automate.exe ui --port 8050
```

Then open **http://127.0.0.1:8050** in your browser to:
- Switch among all **9 benchmark scenarios** (or enter custom STEP file paths).
- **Click faces directly in 3D** on the query part (or in the sidebar list) to toggle selected mating faces.
- Inspect the 3D assembly with highlighted target faces, placed gold $SE(3)$ mated poses, the offscreen 3-panel VTK verification render, and the interactive match table.

### 2. Generate the 9 benchmark STEP assemblies & run all scenarios (CLI batch mode)

```powershell
.venv\Scripts\python.exe scripts\generate_test_assemblies.py
.venv\Scripts\python.exe scripts\run_all_scenarios.py
```

Outputs (JSON match reports + 3-panel 3D PNG renders) are written to `outputs/`.

### 2. Run the automated test suite (14 tests)

```powershell
.venv\Scripts\pytest.exe -v
```

### 3. Inspect face IDs on any STEP part

```powershell
.venv\Scripts\automate.exe inspect --step path\to\part.stp
```

### 4. Run auto-mating on your own STEP part & assembly (e.g. `WM290V_RedrawStart.stp`)

```powershell
.venv\Scripts\automate.exe match `
  --part path\to\chuck_key.stp `
  --faces 13,14,15,16 `
  --assembly path\to\WM290V_RedrawStart.stp `
  --out-dir outputs `
  --name wm290v_chuck_key
```

Optional flags:
- `--one-based`: Interpret `--faces` as 1-based OCCT face indices instead of 0-based.
- `--exact-tol 0.08`: Clearance threshold (mm) separating `EXACT_FIT` from `SLIDING_CLEARANCE_FIT`.
- `--max-clearance 2.5`: Maximum allowed clearance gap (mm) for sliding/clearance mates.
- `--include-self`: Include self-identical part instances if the query part appears in the assembly at the same pose.
- `--no-render`: Export JSON only without VTK PNG rendering.

---

## Project Structure

```text
src/automate/
  models.py       # Data structures, fit types, kinematic pair enums, tolerances
  step_io.py      # XCAF + fallback STEP loader, deterministic B-Rep face indexing
  features.py     # Analytic/freeform surface extraction, outward normals, UV sampling
  matcher.py      # SE(3)-invariant single & pairwise correspondence subgraph search
  pose.py         # Constrained SE(3) solver, BRepExtrema overlap, collision & fit classifier
  visualizer.py   # Offscreen VTK B-Rep renderer + annotated summary PNGs
  cli.py          # `automate` CLI (`inspect` and `match` subcommands)
scripts/
  generate_test_assemblies.py  # Generates 8 multi-part STEP scenarios in scenarios/
  run_all_scenarios.py         # Runs matching & renders all scenarios to outputs/
tests/
  test_automate.py             # 11-test pytest suite (8 scenarios + SE(3) + negative + cross-part)
WRITEUP.md                     # Technical write-up: methodology, semantics, limitations
```
