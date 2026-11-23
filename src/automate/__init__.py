"""SE(3)-invariant B-Rep assembly auto-mating for STEP files."""

from .models import (
    AssemblyData,
    FaceFeature,
    FitType,
    KinematicPairType,
    MatchCandidate,
    MatingConfig,
    PartData,
    SurfaceType,
)
from .step_io import load_step_assembly, load_step_part
from .features import extract_part_features, extract_assembly_features
from .matcher import find_mating_instances
from .visualizer import render_mating_results

__all__ = [
    "AssemblyData",
    "FaceFeature",
    "FitType",
    "KinematicPairType",
    "MatchCandidate",
    "MatingConfig",
    "PartData",
    "SurfaceType",
    "load_step_assembly",
    "load_step_part",
    "extract_part_features",
    "extract_assembly_features",
    "find_mating_instances",
    "render_mating_results",
]
