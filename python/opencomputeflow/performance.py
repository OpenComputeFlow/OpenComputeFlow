"""Explainable analytical estimates for the Phase 0 direct Conv2D path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping

from .contracts import ContractError, Conv2DContract, MappingCandidate, TargetProfile, content_fingerprint


@dataclass(frozen=True)
class PerformanceEstimate:
    model_id: str
    candidate_ref: str
    flops: int
    traffic_bytes: Mapping[str, int]
    compute_cycles: float
    memory_cycles: float
    overhead_cycles: float
    latency_cycles: float
    predicted_bottleneck: str
    confidence: float
    assumptions: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "candidate_ref": self.candidate_ref,
            "flops": self.flops,
            "traffic_bytes": dict(self.traffic_bytes),
            "decomposition": {
                "compute_cycles": self.compute_cycles,
                "memory_cycles": self.memory_cycles,
                "overhead_cycles": self.overhead_cycles,
            },
            "latency_cycles": self.latency_cycles,
            "predicted_bottleneck": self.predicted_bottleneck,
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
        }

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerformanceEstimate":
        if payload.get("schema_version") != 1:
            raise ContractError(f"unsupported PerformanceEstimate schema_version: {payload.get('schema_version')!r}")
        decomposition = payload["decomposition"]
        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            raise ContractError("estimate confidence must be between 0 and 1")
        return cls(
            model_id=payload["model_id"],
            candidate_ref=payload["candidate_ref"],
            flops=payload["flops"],
            traffic_bytes=payload["traffic_bytes"],
            compute_cycles=decomposition["compute_cycles"],
            memory_cycles=decomposition["memory_cycles"],
            overhead_cycles=decomposition["overhead_cycles"],
            latency_cycles=payload["latency_cycles"],
            predicted_bottleneck=payload["predicted_bottleneck"],
            confidence=confidence,
            assumptions=tuple(payload.get("assumptions", ())),
        )


def estimate_direct_conv(
    contract: Conv2DContract,
    candidate: MappingCandidate,
    target: TargetProfile,
) -> PerformanceEstimate:
    """Estimate a legal candidate without pretending to model microarchitecture exactly."""
    candidate.validate(contract, target)
    summary = contract.work_summary()
    performance = target.performance
    flops_per_cycle = float(performance.get("f32_flops_per_cycle", 1.0))
    bandwidth_bytes_per_cycle = float(performance.get("memory_bandwidth_bytes_per_cycle", 16.0))
    overhead_per_tile = float(performance.get("overhead_cycles_per_tile", 8.0))

    n, channels, _, _ = contract.input_shape
    output_channels, _, kernel_h, kernel_w = contract.filter_shape
    _, _, output_h, output_w = contract.output_shape
    stride_h, stride_w = contract.strides
    tile_oc = candidate.tiles["oc"]
    tile_oh = candidate.tiles["oh"]
    tile_ow = candidate.tiles["ow"]
    tiles_n = n
    tiles_oc = math.ceil(output_channels / tile_oc)
    tiles_oh = math.ceil(output_h / tile_oh)
    tiles_ow = math.ceil(output_w / tile_ow)
    tile_count = tiles_n * tiles_oc * tiles_oh * tiles_ow

    traffic = {"input": 0, "filter": 0, "output": 0}
    for batch in range(tiles_n):
        del batch
        for oc_tile in range(tiles_oc):
            actual_oc = min(tile_oc, output_channels - oc_tile * tile_oc)
            for oh_tile in range(tiles_oh):
                actual_oh = min(tile_oh, output_h - oh_tile * tile_oh)
                for ow_tile in range(tiles_ow):
                    actual_ow = min(tile_ow, output_w - ow_tile * tile_ow)
                    input_h = (actual_oh - 1) * stride_h + kernel_h
                    input_w = (actual_ow - 1) * stride_w + kernel_w
                    traffic["input"] += channels * input_h * input_w * 4
                    traffic["filter"] += actual_oc * channels * kernel_h * kernel_w * 4
                    traffic["output"] += actual_oc * actual_oh * actual_ow * 4

    total_traffic = sum(traffic.values())
    compute_cycles = summary["flops"] / flops_per_cycle
    memory_cycles = total_traffic / bandwidth_bytes_per_cycle
    overhead_cycles = tile_count * overhead_per_tile
    latency_cycles = max(compute_cycles, memory_cycles) + overhead_cycles
    if compute_cycles > memory_cycles:
        bottleneck = "compute"
    elif memory_cycles > compute_cycles:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"
    confidence = float(performance.get("model_confidence", 0.0))
    assumptions = (
        "f32 direct convolution",
        "compute and memory phases overlap; overhead is additive",
        "traffic is a tile-level upper-bound and does not model cache conflicts",
    )
    return PerformanceEstimate(
        model_id="analytical-v1",
        candidate_ref=candidate.fingerprint,
        flops=summary["flops"],
        traffic_bytes=traffic,
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        overhead_cycles=overhead_cycles,
        latency_cycles=latency_cycles,
        predicted_bottleneck=bottleneck,
        confidence=max(0.0, min(1.0, confidence)),
        assumptions=assumptions,
    )
