"""Phase 0 reference contracts for OpenComputeFlow."""

from .contracts import (
    ContractError,
    Conv2DContract,
    MappingCandidate,
    TargetProfile,
)
from .evidence import CompilationTrace, DecisionRecord, Measurement, RejectedCandidate
from .performance import PerformanceEstimate, estimate_direct_conv
from .reference import direct_conv2d

__all__ = [
    "ContractError",
    "Conv2DContract",
    "MappingCandidate",
    "TargetProfile",
    "PerformanceEstimate",
    "CompilationTrace",
    "DecisionRecord",
    "Measurement",
    "RejectedCandidate",
    "direct_conv2d",
    "estimate_direct_conv",
]
