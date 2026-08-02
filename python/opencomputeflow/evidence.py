"""Versioned measurement and decision-trace contracts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
import math
import statistics
from typing import Any, Dict, Mapping, Optional, Tuple

from .contracts import ContractError, content_fingerprint


MEASUREMENT_SOURCES = {"real_hardware", "cycle_accurate_model"}


@dataclass(frozen=True)
class Measurement:
    measurement_id: str
    candidate_ref: str
    target_profile_fingerprint: str
    source_kind: str
    metric: str
    unit: str
    collected_at: str
    samples: Tuple[float, ...]
    warmup_runs: int
    environment: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "environment", copy.deepcopy(dict(self.environment)))
        self.validate()

    def validate(self) -> None:
        issues = []
        if self.schema_version != 1:
            issues.append(f"unsupported schema_version {self.schema_version}")
        if not self.measurement_id or not self.candidate_ref or not self.target_profile_fingerprint:
            issues.append("measurement_id, candidate_ref, and target_profile_fingerprint are required")
        if self.source_kind not in MEASUREMENT_SOURCES:
            issues.append("source_kind must be real_hardware or cycle_accurate_model")
        supported_metrics = {"latency_cycles": "cycles", "latency_us": "us"}
        expected_unit = supported_metrics.get(self.metric)
        if expected_unit is None:
            issues.append(f"unsupported measurement metric {self.metric!r}")
        elif self.unit != expected_unit:
            issues.append(f"metric {self.metric!r} requires unit {expected_unit!r}")
        try:
            collected_at = datetime.fromisoformat(self.collected_at.replace("Z", "+00:00"))
            if collected_at.tzinfo is None:
                issues.append("collected_at must include a timezone")
        except (AttributeError, ValueError):
            issues.append("collected_at must be an ISO-8601 timestamp")
        if not self.samples:
            issues.append("at least one sample is required")
        elif any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in self.samples):
            issues.append("samples must be non-negative numbers")
        if isinstance(self.warmup_runs, bool) or not isinstance(self.warmup_runs, int) or self.warmup_runs < 0:
            issues.append("warmup_runs must be a non-negative integer")
        required_environment = {"device_id", "backend_version", "clock_policy", "threads"}
        missing = required_environment - set(self.environment)
        if missing:
            issues.append(f"environment is missing fields: {sorted(missing)}")
        threads = self.environment.get("threads")
        if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0):
            issues.append("environment.threads must be a positive integer")
        if issues:
            raise ContractError("invalid measurement: " + "; ".join(issues))

    @property
    def summary(self) -> Dict[str, float]:
        ordered = sorted(float(value) for value in self.samples)
        p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
        return {
            "min": ordered[0],
            "median": float(statistics.median(ordered)),
            "p90": ordered[p90_index],
            "max": ordered[-1],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "measurement_id": self.measurement_id,
            "candidate_ref": self.candidate_ref,
            "target_profile_fingerprint": self.target_profile_fingerprint,
            "source_kind": self.source_kind,
            "metric": self.metric,
            "unit": self.unit,
            "collected_at": self.collected_at,
            "warmup_runs": self.warmup_runs,
            "samples": list(self.samples),
            "summary": self.summary,
            "environment": copy.deepcopy(dict(self.environment)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Measurement":
        measurement = cls(
            measurement_id=payload["measurement_id"],
            candidate_ref=payload["candidate_ref"],
            target_profile_fingerprint=payload["target_profile_fingerprint"],
            source_kind=payload["source_kind"],
            metric=payload["metric"],
            unit=payload["unit"],
            collected_at=payload["collected_at"],
            samples=payload["samples"],
            warmup_runs=payload["warmup_runs"],
            environment=payload["environment"],
            schema_version=payload.get("schema_version"),
        )
        declared_summary = payload.get("summary")
        if declared_summary is not None and declared_summary != measurement.summary:
            raise ContractError("measurement summary does not match samples")
        return measurement

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())


@dataclass(frozen=True)
class RejectedCandidate:
    candidate_ref: str
    reason_code: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "candidate_ref": self.candidate_ref,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RejectedCandidate":
        return cls(
            candidate_ref=payload["candidate_ref"],
            reason_code=payload["reason_code"],
            detail=payload["detail"],
        )


@dataclass(frozen=True)
class DecisionRecord:
    site: str
    candidates_generated: Tuple[str, ...]
    candidates_legal: Tuple[str, ...]
    selected_ref: str
    estimate_refs: Mapping[str, str]
    rejected: Tuple[RejectedCandidate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates_generated", tuple(self.candidates_generated))
        object.__setattr__(self, "candidates_legal", tuple(self.candidates_legal))
        object.__setattr__(self, "estimate_refs", dict(self.estimate_refs))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        self.validate()

    def validate(self) -> None:
        generated = set(self.candidates_generated)
        legal = set(self.candidates_legal)
        rejected = {item.candidate_ref for item in self.rejected}
        issues = []
        if not self.site:
            issues.append("site is required")
        if len(generated) != len(self.candidates_generated):
            issues.append("candidates_generated must not contain duplicates")
        if not legal <= generated:
            issues.append("legal candidates must be generated candidates")
        if rejected - generated:
            issues.append("rejected candidates must be generated candidates")
        if legal & rejected:
            issues.append("a candidate cannot be both legal and rejected")
        if self.selected_ref not in legal:
            issues.append("selected_ref must identify a legal candidate")
        if set(self.estimate_refs) != legal:
            issues.append("estimate_refs must contain exactly the legal candidates")
        if issues:
            raise ContractError("invalid decision record: " + "; ".join(issues))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site": self.site,
            "candidates_generated": list(self.candidates_generated),
            "candidates_legal": list(self.candidates_legal),
            "selected_ref": self.selected_ref,
            "estimate_refs": dict(self.estimate_refs),
            "rejected": [item.to_dict() for item in self.rejected],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionRecord":
        return cls(
            site=payload["site"],
            candidates_generated=payload["candidates_generated"],
            candidates_legal=payload["candidates_legal"],
            selected_ref=payload["selected_ref"],
            estimate_refs=payload["estimate_refs"],
            rejected=tuple(RejectedCandidate.from_dict(item) for item in payload.get("rejected", ())),
        )


@dataclass(frozen=True)
class CompilationTrace:
    trace_id: str
    input_fingerprint: str
    target_profile_ref: str
    target_profile_fingerprint: str
    backend_version: str
    cost_model_id: str
    calibration_id: Optional[str]
    decisions: Tuple[DecisionRecord, ...]
    measurement_refs: Tuple[str, ...] = ()
    pipeline_version: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "measurement_refs", tuple(self.measurement_refs))
        self.validate()

    def validate(self) -> None:
        issues = []
        if self.schema_version != 1 or self.pipeline_version <= 0:
            issues.append("schema_version must be 1 and pipeline_version must be positive")
        required = (
            self.trace_id,
            self.input_fingerprint,
            self.target_profile_ref,
            self.target_profile_fingerprint,
            self.backend_version,
            self.cost_model_id,
        )
        if any(not value for value in required):
            issues.append("trace identity, target, backend, and model fields are required")
        if not self.decisions:
            issues.append("at least one decision is required")
        if len(set(self.measurement_refs)) != len(self.measurement_refs):
            issues.append("measurement_refs must not contain duplicates")
        if issues:
            raise ContractError("invalid compilation trace: " + "; ".join(issues))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "pipeline_version": self.pipeline_version,
            "input_fingerprint": self.input_fingerprint,
            "target_profile_ref": self.target_profile_ref,
            "target_profile_fingerprint": self.target_profile_fingerprint,
            "backend_version": self.backend_version,
            "cost_model": {
                "id": self.cost_model_id,
                "calibration_id": self.calibration_id,
            },
            "decisions": [item.to_dict() for item in self.decisions],
            "measurement_refs": list(self.measurement_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompilationTrace":
        cost_model = payload["cost_model"]
        return cls(
            trace_id=payload["trace_id"],
            input_fingerprint=payload["input_fingerprint"],
            target_profile_ref=payload["target_profile_ref"],
            target_profile_fingerprint=payload["target_profile_fingerprint"],
            backend_version=payload["backend_version"],
            cost_model_id=cost_model["id"],
            calibration_id=cost_model.get("calibration_id"),
            decisions=tuple(DecisionRecord.from_dict(item) for item in payload["decisions"]),
            measurement_refs=payload.get("measurement_refs", ()),
            pipeline_version=payload["pipeline_version"],
            schema_version=payload.get("schema_version"),
        )

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())
