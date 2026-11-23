from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np


class SurfaceType(str, Enum):
    PLANE = "PLANE"
    CYLINDER = "CYLINDER"
    CONE = "CONE"
    SPHERE = "SPHERE"
    TORUS = "TORUS"
    FREEFORM = "FREEFORM"


class FitType(str, Enum):
    EXACT_FIT = "EXACT_FIT"
    INTERFERENCE_FIT = "INTERFERENCE_FIT"
    SLIDING_CLEARANCE_FIT = "SLIDING_CLEARANCE_FIT"


class KinematicPairType(str, Enum):
    RIGID_0DOF = "RIGID_0DOF"
    PRISMATIC_SLIDE = "PRISMATIC_SLIDE"
    REVOLUTE = "REVOLUTE"
    CYLINDRICAL = "CYLINDRICAL"
    PLANAR = "PLANAR"
    SPHERICAL = "SPHERICAL"
    GENERAL_CONSTRAINED = "GENERAL_CONSTRAINED"


@dataclass
class FaceFeature:
    """Geometric and topological descriptor for a single B-Rep face."""

    global_id: int
    part_id: int
    part_name: str
    local_face_id: int
    surface_type: SurfaceType
    area: float
    centroid: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)

    # Sense: +1 = convex (outer shaft/ball/cone), -1 = concave (bore/pocket/groove), 0 = plane/neutral
    sense: int = 0

    # Analytic surface parameters (in world/assembly coordinates)
    normal: Optional[np.ndarray] = None  # unit outward normal for PLANE (or mean normal)
    origin: Optional[np.ndarray] = None  # point on plane/axis, or sphere center, or cone apex
    axis: Optional[np.ndarray] = None    # unit axis for CYLINDER / CONE / TORUS
    radius: Optional[float] = None       # CYLINDER / SPHERE / mean CONE radius
    r_min: Optional[float] = None        # CONE min radius
    r_max: Optional[float] = None        # CONE max radius
    half_angle: Optional[float] = None   # CONE semi-angle (radians, >0)
    major_radius: Optional[float] = None # TORUS major radius
    minor_radius: Optional[float] = None # TORUS minor radius

    # Trimmed face extents
    axial_span: Optional[float] = None   # span along cylinder/cone axis
    angular_span: Optional[float] = None # radians in [0, 2*pi]
    extent_u: Optional[float] = None     # planar PCA major extent
    extent_v: Optional[float] = None     # planar PCA minor extent

    # Freeform / curvature summary
    mean_curvature: float = 0.0
    gaussian_curvature: float = 0.0
    pca_axes: Optional[np.ndarray] = None  # (3, 3) rows = principal directions

    # Stratified interior samples strictly inside the trimmed B-Rep face
    sample_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sample_normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # Neighboring local_face_ids on the same part
    adjacent_local_ids: tuple[int, ...] = ()

    # Raw OCCT face handle (not serialized)
    occt_face: Any = field(default=None, repr=False)


@dataclass
class PartData:
    """A single solid/body instance extracted from a STEP file."""

    part_id: int
    name: str
    shape: Any  # TopoDS_Shape in assembly world coordinates
    faces: list[FaceFeature] = field(default_factory=list)
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    volume: float = 0.0


@dataclass
class AssemblyData:
    """Loaded STEP assembly containing multiple parts."""

    source_path: str
    parts: list[PartData] = field(default_factory=list)

    @property
    def all_faces(self) -> list[FaceFeature]:
        faces: list[FaceFeature] = []
        for p in self.parts:
            faces.extend(p.faces)
        return faces


@dataclass
class MatingConfig:
    """Tolerances and matching parameters (units in mm and degrees)."""

    # Angle tolerances
    angle_tol_deg: float = 4.0
    parallel_tol_deg: float = 3.0

    # Gap / clearance tolerances
    interference_tol_mm: float = 0.15      # max allowed solid penetration (negative gap)
    exact_fit_tol_mm: float = 0.08         # |gap| <= this -> EXACT_FIT
    max_clearance_mm: float = 2.5          # max gap for SLIDING_CLEARANCE_FIT
    relative_clearance_ratio: float = 0.08 # or up to 8% of radius/width for larger parts

    # Extent / overlap checks
    min_coverage: float = 0.45             # min fraction of query or target face overlapping
    edge_overlap_tol_mm: float = 1.5       # tolerance for trimmed boundary projection
    transverse_overflow_tol_mm: float = 1.5

    # Geometric invariance tolerances
    axis_dist_tol_mm: float = 0.75
    cone_angle_tol_deg: float = 3.0
    curvature_rel_tol: float = 0.35

    # Search controls
    allow_cross_part_matches: bool = True
    max_cross_part_gap_mm: float = 5.0
    check_solid_collision: bool = True
    collision_probe_depth_mm: float = 0.35
    max_candidates_per_face: int = 250


@dataclass
class MatchCandidate:
    """A verified match instance in the assembly for the selected query faces."""

    match_id: int
    part_ids: list[int]
    part_names: list[str]
    query_face_ids: list[int]          # local face IDs on query part
    target_global_face_ids: list[int]  # global face IDs in assembly
    target_local_face_ids: list[int]   # local face IDs within their parts
    target_part_ids: list[int]         # part_id for each matched target face

    transform_4x4: np.ndarray          # SE(3) matrix aligning query -> target
    fit_type: FitType
    kinematic_pair: KinematicPairType
    dof_translation: int
    dof_rotation: int

    mean_gap_mm: float
    min_gap_mm: float
    max_gap_mm: float
    opposing_clearance_mm: Optional[float]
    normal_error_deg: float
    overlap_score: float
    score: float                       # overall quality score (higher = better)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "part_ids": self.part_ids,
            "part_names": self.part_names,
            "query_face_ids": self.query_face_ids,
            "target_global_face_ids": self.target_global_face_ids,
            "target_local_face_ids": self.target_local_face_ids,
            "target_part_ids": self.target_part_ids,
            "fit_type": self.fit_type.value,
            "kinematic_pair": self.kinematic_pair.value,
            "dof_translation": self.dof_translation,
            "dof_rotation": self.dof_rotation,
            "mean_gap_mm": round(float(self.mean_gap_mm), 4),
            "min_gap_mm": round(float(self.min_gap_mm), 4),
            "max_gap_mm": round(float(self.max_gap_mm), 4),
            "opposing_clearance_mm": (
                round(float(self.opposing_clearance_mm), 4)
                if self.opposing_clearance_mm is not None
                else None
            ),
            "normal_error_deg": round(float(self.normal_error_deg), 3),
            "overlap_score": round(float(self.overlap_score), 4),
            "score": round(float(self.score), 4),
            "transform_4x4": np.round(self.transform_4x4, 6).tolist(),
            "reasoning": self.reasoning,
        }
