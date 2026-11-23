# Technical Write-Up: SE(3)-Invariant B-Rep Auto-Mating

## 1. Approach & Architectural Justification

Assembly mating in CAD is fundamentally a **constrained subgraph isomorphism + $SE(3)$ kinematic alignment** problem over B-Rep surfaces. Rather than relying on mesh-only point clouds (which lose exact radii, cone half-angles, and outward surface orientation) or hardcoded feature templates (like "circular hole" or "square peg"), we operate directly on **exact Open CASCADE B-Rep topology and analytic geometry** augmented with trimmed-interior point/normal sampling.

Why this hybrid B-Rep + sampled-interior approach works best:
1. **Exact surface semantics**: B-Rep faces directly expose surface class (`Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`, `BSpline/Freeform`), exact curvature/radius/half-angle, and outward material orientation (`TopAbs_REVERSED`).
2. **Pose invariance without brittle frame heuristics**: Comparing pairwise invariants (dot products of normals/axes, signed plane-to-plane offsets, half-space overlap intervals, axis-to-axis line distances, chirality determinants) makes candidate search completely invariant to rigid transforms $T \in SE(3)$.
3. **Handles partial & asymmetric face contacts**: A short sliding jaw on long vice ways, a $180^\circ$ V-belt arc on a $360^\circ$ pulley, or a $16\text{ mm}$ chuck key entering a $12\text{ mm}$ pocket do **not** have equal surface areas. By solving the constrained $SE(3)$ alignment first and then evaluating exact trimmed-face overlap via `BRepExtrema_DistShapeShape`, asymmetric contacts succeed cleanly while impossible placements are rejected.

---

## 2. Geometric & Semantic Properties Matched On

### 2.1 Single-Face Compatibility
For a query face $q$ to mate against a target face $t$:
- **Surface Type & Material Sense**:
  - `PLANE` $\leftrightarrow$ `PLANE`: any planar face (outward normals must oppose after alignment).
  - `CYLINDER` $\leftrightarrow$ `CYLINDER`, `SPHERE` $\leftrightarrow$ `SPHERE`, `CONE` $\leftrightarrow$ `CONE`, `TORUS` $\leftrightarrow$ `TORUS`: must have **opposite radial sense** (`q.sense * t.sense == -1`, i.e., convex shaft/ball/flank into concave bore/socket/groove).
  - `CONE` $\leftrightarrow$ `CONE`: half-angles must match within `cone_angle_tol_deg` ($3^\circ$), and radial spans $[r_{\min}, r_{\max}]$ must overlap within clearance.
  - `FREEFORM` $\leftrightarrow$ `FREEFORM`: compatible curvature sign (`mean_curvature`) and comparable surface area.
- **Scale-Aware Size Clearance**:
  - For curved surfaces, radial clearance $c = R_{\text{concave}} - R_{\text{convex}}$ must satisfy $-\delta_{\text{int}} \le c \le c_{\text{eff}}$, where $c_{\text{eff}} = \min(c_{\max}, \max(0.35\text{ mm}, 0.08 \cdot D))$. This prevents a $10\text{ mm}$ shaft from matching a $14\text{ mm}$ hole while allowing realistic sliding clearances on larger parts.

### 2.2 Pairwise $SE(3)$-Invariant Group Constraints
Matching faces independently causes massive false positives. For every pair $(q_i, q_j)$ and candidate $(t_i, t_j)$:
1. **Directional Angles**:
   - Planar/freeform normals negate upon mating ($R\mathbf{n}_q = -\mathbf{n}_t$), whereas canonical cone widening axes preserve direction ($R\mathbf{a}_q = +\mathbf{a}_t$). Pairwise dot products are checked against expected sign parity.
2. **Signed Parallel-Plane Separation (Male vs. Female Discrimination)**:
   - For parallel opposing planes ($\mathbf{n}_1 \cdot \mathbf{n}_2 \approx -1$), we compute $d_q = (\mathbf{o}_{q_2} - \mathbf{o}_{q_1}) \cdot \mathbf{n}_{q_1}$ and $d_t = (\mathbf{o}_{t_2} - \mathbf{o}_{t_1}) \cdot \mathbf{n}_{t_1}$.
   - Because male parts have outward normals pointing *away* from each other ($d_q = -W_{\text{key}}$) and female slots have outward normals pointing *toward* each other ($d_t = +W_{\text{slot}}$), the **invariant opposing clearance** is:
     $$g_{ij} = d_t + d_q = W_{\text{slot}} - W_{\text{key}} \in [-\delta_{\text{int}}, c_{\text{eff}}]$$
   - Attempting to match a male key against a male block gives $g_{ij} = -2W < -\delta_{\text{int}}$, immediately eliminating convex-to-convex false positives.
3. **Non-Parallel Plane Half-Space Overlap**:
   - Two non-parallel planes on a convex edge vs. a concave pocket have identical dihedral angles. Checking the signed projection intervals of each face's interior points onto the other's outward normal kills convex-to-convex and concave-to-concave false matches.
4. **3D Chirality Determinant**:
   - For any non-coplanar triple of oriented normals/axes, $\det([\mathbf{v}_{t_1}, \mathbf{v}_{t_2}, \mathbf{v}_{t_3}])$ must satisfy proper $SO(3)$ parity ($\det(-R) = -1$ for 3 planes), rejecting mirror-image false positives.
5. **Axis-to-Axis & Plane-to-Axis Distances**:
   - Shortest 3D line distance between cylinder/cone axes and signed perpendicular offset between parallel keyway planes and shaft/bore axes.

### 2.3 Constrained $SE(3)$ Solver & Fit Classification
- **Rotation $R \in SO(3)$**: Solved via Kabsch/Wahba SVD when $\ge 2$ non-collinear directions exist; when rank $= 1$ (e.g., single cylinder, coaxial cones, or parallel opposing planes), a 1-DOF angular scan optimizes overlap around the free axis. For `FREEFORM` faces, all 4 right-handed sign symmetries of the 3D PCA principal frame are refined with 4 iterations of joint point+normal Kabsch ICP.
- **Translation $\mathbf{t} \in \mathbb{R}^3$ & 1-DOF Blind-Pocket Seating**: Solved via rank-revealing SVD linear least squares (`CYLINDER`/`CONE`/`SPHERE` constraints weighted $50\times$ higher than `PLANE` constraints so concentric shafts stay centered; `FREEFORM` faces contribute subsampled tangent-plane equations across the surface). Along a 1-DOF sliding axis (`dof_t == 1`), when the query face span exceeds a blind pocket's depth (e.g. a $16\text{ mm}$ chuck key tip entering a $12\text{ mm}$ deep pinion socket), end-flush axial shifts are evaluated so the tip seats cleanly above the pocket floor.
- **Full-Body Solid Non-Penetration & Co-Surface Seam Merging**:
  - In addition to probing overlapping points on the selected faces, `_check_solid_collision` probes interior material points ($\mathbf{p}_{\text{surf}} - 0.20\text{ mm}\cdot\mathbf{n}_{\text{surf}}$) on all **unselected faces** of the query part against the target solid (`BRepClass3d_SolidClassifier`). This eliminates upside-down $D_4$ permutations (such as a chuck key with its T-handle pointing inward into the solid chuck body).
  - Adjacent seam-split target faces sharing identical surface equations (e.g., two $180^\circ$ half-cylinder bore faces) are automatically expanded and deduplicated into one unified match (`_expand_and_deduplicate_cosurface_siblings`).
- **Exact vs. Sliding/Clearance Classification**:
  - Evaluates exact analytic surface normal opposition, signed surface gap $s(\mathbf{p}')$ along $\mathbf{n}_t$, invariant opposing clearance `opp_gap`, and trimmed B-Rep overlap (`BRepExtrema_DistShapeShape`).
  - `INTERFERENCE_FIT`: $s_{\min} < -0.01\text{ mm}$ (within $-\delta_{\text{int}}$).
  - `EXACT_FIT`: `gap_indicator <= exact_fit_tol_mm` ($0.08\text{ mm}$) and `max_face_mean_gap <= exact_fit_tol_mm + 0.05`.
  - `SLIDING_CLEARANCE_FIT`: controlled positive gap (`> 0.08 mm` up to $c_{\text{eff}}$), where opposing walls/cylinders/dovetail cheeks stay parallel or coaxial without coinciding.

---

## 3. Evaluation Summary

All 9 synthetic multi-part STEP scenarios and 5 additional robustness tests (`14/14` in `pytest`) pass:

| Scenario | Selected Faces | Expected | Found | Fit & Kinematic Pair | Time |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `chuck_key_3slot` | 4 square key tip planes | 3 | **3** | `EXACT_FIT` (`PRISMATIC_SLIDE`), gap `0.010mm` | 3.01s |
| `chuck_key_4slot` | 4 square key tip planes | 4 | **4** | `EXACT_FIT` (`PRISMATIC_SLIDE`), gap `0.010mm` | 3.93s |
| `bench_vice_sliding_jaw` | 3 tongue guide planes | 1 | **1** | `SLIDING_CLEARANCE_FIT` (`PRISMATIC_SLIDE`), opp_clear `0.500mm` | 1.16s |
| `keyed_shaft_to_hubs` | Shaft cyl + 3 key planes | 2 | **2** | `EXACT_FIT` (`PRISMATIC_SLIDE`), gap `0.017mm`, opp_clear `0.030mm` | 0.75s |
| `pulley_on_motor_shaft` | Bore cyl + shoulder plane | 1 | **1** | `SLIDING_CLEARANCE_FIT` (`REVOLUTE`), opp_clear `0.150mm` | 0.50s |
| `vbelt_on_vpulley` | 2 V-belt cones + inner cyl | 1 | **1** | `EXACT_FIT` (`REVOLUTE`), gap `0.028mm` | 1.22s |
| `spherical_ball_joint` | 1 ball sphere | 2 | **2** | `EXACT_FIT` (`SPHERICAL`), opp_clear `0.020mm` | 0.58s |
| `freeform_bspline_mate` | 1 BSpline loft face | 1 | **1** | `EXACT_FIT` (`RIGID_0DOF`), gap `0.000mm` | 0.78s |
| `dovetail_cross_slide` | 2 $60^\circ$ cheeks + bottom | 1 | **1** | `SLIDING_CLEARANCE_FIT` (`PRISMATIC_SLIDE`), gap `0.206mm` | 0.65s |

Additional verified tests:
- **Explicit $SE(3)$ invariance & outward handle verification**: Applying an arbitrary rotation + large translation `(-450, 320, -180)` to the chuck key produces the exact same 3 pinion matches with all 3 T-handles oriented radially outward.
- **Freeform `RIGID_0DOF` & $SE(3)$ rotation invariance**: Rotating the BSpline cam plug around an arbitrary 3D axis recovers the exact socket match with `RIGID_0DOF (T:0, R:0)`.
- **Seam-split half-cylinder bore merging**: A housing whose cylindrical bore consists of two sewn $180^\circ$ half-cylinder B-Rep faces merges both halves into 1 unified match (`target_local_face_ids` containing both half-cylinders).
- **Negative decoy rejection**: An oversized `10.5mm` square key returns `0` matches against `8.02mm` sockets and `11.5mm` decoy sockets.
- **Cross-part matching**: A slider block matches a channel formed across 3 separate bolted parts (`BasePlate`, `LeftGuideRail`, `RightGuideRail`) as `SLIDING_CLEARANCE_FIT`.

---

## 4. What Breaks & Limitations

1. **Multi-face split on the query side vs. unsplit target**:
   - Co-surface sibling expansion currently merges split faces on the **target** part (e.g. a 1-face query shaft matching a 2-face split target bore). If the **query** selection itself splits one cylinder into two selected half-cylinder faces while the target bore is a single $360^\circ$ face, the injective (`1-to-1`) query-to-target assignment would need pre-merging on the query side as well.
2. **Target-part vs. full-assembly collision**:
   - `_check_solid_collision` checks the unselected query part body against the matched target part(s), which prevents upside-down insertions into the target part itself, but does not yet test collision against *unrelated third-party parts* elsewhere in the assembly.
3. **Large assemblies ($10{,}000+$ faces) & large query groups ($k > 12$)**:
   - Backtracking search runs in $0.5\text{s}$–$3.9\text{s}$ on typical parts, but selecting 20+ faces on a 500-part assembly without spatial hashing on anchor pairs could hit combinatorial slowdown.
4. **`WM290V_RedrawStart.stp` availability**:
   - The GrabCAD file `WM290V_RedrawStart.stp` requires authenticated GrabCAD login to download and was not included locally in the workspace, so verification was performed against 9 synthetic STEP assemblies (including 3-slot and 4-slot lathe chucks modeled after the assignment) while keeping the CLI directly compatible with `WM290V_RedrawStart.stp`.

---

## 5. What I Would Do With Another Week

1. **Co-surface B-Rep Face Merging (Canonical Region Grouping)**:
   - Merge adjacent split faces sharing the same underlying analytic plane/cylinder/cone/sphere equation (e.g., two $180^\circ$ seam-split bore halves) before matching so CAD exporter face splitting never breaks 1-to-1 correspondences.
2. **Point-to-Plane ICP + Surface Curvature Signatures for Freeform Faces**:
   - Replace the mean-normal approximation on `FREEFORM` faces with FPFH / curvature-histogram descriptors and Gauss-Newton point-to-plane ICP (plus full $6\times 6$ twist-wrench Jacobian rank analysis for exact freeform DOF classification).
3. **Full-Assembly Interference Sweeps along Kinematic DOFs**:
   - After finding a candidate mate pose $T$, run a fast BVH/OCCT solid intersection check on the **entire query part** against **all surrounding assembly parts** (and along the 1-DOF insertion/sliding ray, verifying unobstructed approach path for keys and jaws).
4. **Geometric Hashing / Spatial Indexing**:
   - Index target face pairs into a hash table quantized by `(surface_types, angle_bin, distance_bin)` for sub-second query time even on full 500-part machine-tool assemblies.
