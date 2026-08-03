"""Versioned, dependency-free contracts used by the Phase 0 reference path."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
F32_BYTES = 4
CONV_AXES = ("n", "oc", "oh", "ow", "ic", "kh", "kw")


class ContractError(ValueError):
    """Raised when a contract or mapping violates its declared invariants."""


def _canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("non-finite numbers cannot be fingerprinted")
        if value == 0.0:
            return "0"
        if value.is_integer():
            return str(int(value))
        return repr(value).lower()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError("canonical JSON object keys must be strings")
        return "{" + ",".join(
            _canonical_json(key) + ":" + _canonical_json(value[key])
            for key in sorted(value)
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    raise ContractError(f"unsupported canonical JSON value: {type(value).__name__}")


def content_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _int_tuple(value: Sequence[int], size: int, name: str) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ContractError(f"{name} must contain exactly {size} integers")
    try:
        actual_size = len(value)
    except TypeError as error:
        raise ContractError(f"{name} must contain exactly {size} integers") from error
    if actual_size != size:
        raise ContractError(f"{name} must contain exactly {size} integers")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise ContractError(f"{name} must contain exactly {size} integers")
    return result


def _positive_tuple(value: Sequence[int], size: int, name: str) -> Tuple[int, ...]:
    result = _int_tuple(value, size, name)
    if any(item <= 0 for item in result):
        raise ContractError(f"{name} must contain positive integers")
    return result


@dataclass(frozen=True)
class Conv2DContract:
    """The restricted f32 direct Conv2D semantic contract from the MVP."""

    input_shape: Tuple[int, int, int, int]
    filter_shape: Tuple[int, int, int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    input_layout: str = "NCHW"
    filter_layout: str = "OIHW"
    output_layout: str = "NCHW"
    input_dtype: str = "f32"
    filter_dtype: str = "f32"
    output_dtype: str = "f32"
    accumulation_dtype: str = "f32"
    mode: str = "forward_inference"
    algorithm: str = "direct"
    post_ops: Tuple[str, ...] = ()
    allow_fma_contraction: bool = True
    allow_reassociation: bool = False
    deterministic: bool = True
    absolute_tolerance: float = 1.0e-5
    relative_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", _positive_tuple(self.input_shape, 4, "input_shape"))
        object.__setattr__(self, "filter_shape", _positive_tuple(self.filter_shape, 4, "filter_shape"))
        object.__setattr__(self, "strides", _positive_tuple(self.strides, 2, "strides"))
        object.__setattr__(self, "dilation", _positive_tuple(self.dilation, 2, "dilation"))
        object.__setattr__(self, "padding", _int_tuple(self.padding, 4, "padding"))
        object.__setattr__(self, "post_ops", tuple(self.post_ops))
        self.validate()

    @property
    def output_shape(self) -> Tuple[int, int, int, int]:
        _, _, height, width = self.input_shape
        output_channels, _, kernel_h, kernel_w = self.filter_shape
        stride_h, stride_w = self.strides
        dilation_h, dilation_w = self.dilation
        pad_top, pad_bottom, pad_left, pad_right = self.padding
        effective_h = dilation_h * (kernel_h - 1) + 1
        effective_w = dilation_w * (kernel_w - 1) + 1
        output_h = (height + pad_top + pad_bottom - effective_h) // stride_h + 1
        output_w = (width + pad_left + pad_right - effective_w) // stride_w + 1
        return (self.input_shape[0], output_channels, output_h, output_w)

    def validate(self) -> None:
        issues = []
        input_channels = self.input_shape[1]
        output_channels, filter_channels, _, _ = self.filter_shape
        if any(item < 0 for item in self.padding):
            issues.append("padding must be non-negative")
        if self.groups != 1:
            issues.append("groups must be 1 in the Phase 0 Conv2D contract")
        if self.dilation != (1, 1):
            issues.append("dilation must be (1, 1) in the Phase 0 Conv2D contract")
        if input_channels != filter_channels:
            issues.append("filter input channels must equal input channels")
        if self.input_layout != "NCHW" or self.filter_layout != "OIHW" or self.output_layout != "NCHW":
            issues.append("Phase 0 layouts must be input/output NCHW and filter OIHW")
        if any(dtype != "f32" for dtype in (
            self.input_dtype,
            self.filter_dtype,
            self.output_dtype,
            self.accumulation_dtype,
        )):
            issues.append("Phase 0 Conv2D supports f32 storage and accumulation only")
        if self.mode != "forward_inference":
            issues.append("mode must be forward_inference")
        if self.algorithm != "direct":
            issues.append("Phase 0 algorithm must be direct")
        if self.post_ops:
            issues.append("Phase 0 does not support post-ops")
        for name, value in (
            ("allow_fma_contraction", self.allow_fma_contraction),
            ("allow_reassociation", self.allow_reassociation),
            ("deterministic", self.deterministic),
        ):
            if not isinstance(value, bool):
                issues.append(f"{name} must be boolean")
        for name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                issues.append(f"{name} must be non-negative")
        output_shape = self.output_shape
        if output_shape[2] <= 0 or output_shape[3] <= 0:
            issues.append("computed output spatial dimensions must be positive")
        if issues:
            raise ContractError("invalid Conv2D contract: " + "; ".join(issues))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "op": "conv2d",
            "input_shape": list(self.input_shape),
            "filter_shape": list(self.filter_shape),
            "output_shape": list(self.output_shape),
            "strides": list(self.strides),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "groups": self.groups,
            "input_layout": self.input_layout,
            "filter_layout": self.filter_layout,
            "output_layout": self.output_layout,
            "input_dtype": self.input_dtype,
            "filter_dtype": self.filter_dtype,
            "output_dtype": self.output_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "mode": self.mode,
            "algorithm": self.algorithm,
            "post_ops": list(self.post_ops),
            "numerical_policy": {
                "allow_fma_contraction": self.allow_fma_contraction,
                "allow_reassociation": self.allow_reassociation,
                "deterministic": self.deterministic,
                "absolute_tolerance": self.absolute_tolerance,
                "relative_tolerance": self.relative_tolerance,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Conv2DContract":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ContractError(f"unsupported Conv2D schema_version: {payload.get('schema_version')!r}")
        if payload.get("op") != "conv2d":
            raise ContractError("contract op must be conv2d")
        numerical_policy = payload.get("numerical_policy", {})
        contract = cls(
            input_shape=payload["input_shape"],
            filter_shape=payload["filter_shape"],
            strides=payload.get("strides", (1, 1)),
            padding=payload.get("padding", (0, 0, 0, 0)),
            dilation=payload.get("dilation", (1, 1)),
            groups=payload.get("groups", 1),
            input_layout=payload.get("input_layout", "NCHW"),
            filter_layout=payload.get("filter_layout", "OIHW"),
            output_layout=payload.get("output_layout", "NCHW"),
            input_dtype=payload.get("input_dtype", "f32"),
            filter_dtype=payload.get("filter_dtype", "f32"),
            output_dtype=payload.get("output_dtype", "f32"),
            accumulation_dtype=payload.get("accumulation_dtype", "f32"),
            mode=payload.get("mode", "forward_inference"),
            algorithm=payload.get("algorithm", "direct"),
            post_ops=payload.get("post_ops", ()),
            allow_fma_contraction=numerical_policy.get("allow_fma_contraction", True),
            allow_reassociation=numerical_policy.get("allow_reassociation", False),
            deterministic=numerical_policy.get("deterministic", True),
            absolute_tolerance=numerical_policy.get("absolute_tolerance", 1.0e-5),
            relative_tolerance=numerical_policy.get("relative_tolerance", 1.0e-5),
        )
        declared_output = payload.get("output_shape")
        if declared_output is not None and tuple(declared_output) != contract.output_shape:
            raise ContractError(
                f"declared output_shape {tuple(declared_output)} does not match inferred {contract.output_shape}"
            )
        return contract

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())

    def work_summary(self) -> Dict[str, Any]:
        n, channels, height, width = self.input_shape
        output_channels, _, kernel_h, kernel_w = self.filter_shape
        _, _, output_h, output_w = self.output_shape
        multiply_accumulates = n * output_channels * output_h * output_w * channels * kernel_h * kernel_w
        return {
            "output_elements": n * output_channels * output_h * output_w,
            "multiply_accumulates": multiply_accumulates,
            "flops": 2 * multiply_accumulates,
            "logical_bytes": {
                "input": n * channels * height * width * F32_BYTES,
                "filter": output_channels * channels * kernel_h * kernel_w * F32_BYTES,
                "output": n * output_channels * output_h * output_w * F32_BYTES,
            },
        }


@dataclass(frozen=True)
class TargetProfile:
    target_id: str
    backend: str
    profile_version: int = 1
    features: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    performance: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", copy.deepcopy(dict(self.features)))
        object.__setattr__(
            self,
            "resources",
            {str(key): copy.deepcopy(dict(value)) for key, value in self.resources.items()},
        )
        object.__setattr__(self, "performance", copy.deepcopy(dict(self.performance)))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        self.validate()

    @property
    def target_ref(self) -> str:
        return f"{self.target_id}@{self.profile_version}"

    def validate(self) -> None:
        issues = []
        if self.schema_version != SCHEMA_VERSION:
            issues.append(f"unsupported schema_version {self.schema_version}")
        if not self.target_id or not self.backend:
            issues.append("target_id and backend are required")
        if self.profile_version <= 0:
            issues.append("profile_version must be positive")
        if not self.resources:
            issues.append("at least one target resource is required")
        for resource_id, resource in self.resources.items():
            if not resource.get("kind"):
                issues.append(f"resource {resource_id!r} must declare kind")
            capacity = resource.get("capacity_bytes")
            if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0):
                issues.append(f"resource {resource_id!r} capacity_bytes must be positive")
            explicitly_managed = resource.get("explicitly_managed")
            if explicitly_managed is not None and not isinstance(explicitly_managed, bool):
                issues.append(f"resource {resource_id!r} explicitly_managed must be boolean")
        vector = self.features.get("vector", {})
        if vector and not isinstance(vector.get("supports_masked_ops", False), bool):
            issues.append("features.vector.supports_masked_ops must be boolean")
        positive_metrics = {
            "f32_flops_per_cycle",
            "memory_bandwidth_bytes_per_cycle",
            "overhead_cycles_per_tile",
        }
        unknown_metrics = set(self.performance) - positive_metrics - {"model_confidence"}
        if unknown_metrics:
            issues.append(f"unknown performance fields: {sorted(unknown_metrics)}")
        for name in positive_metrics & set(self.performance):
            value = self.performance[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                issues.append(f"performance.{name} must be positive")
        confidence = self.performance.get("model_confidence")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0
        ):
            issues.append("performance.model_confidence must be between 0 and 1")
        if issues:
            raise ContractError("invalid target profile: " + "; ".join(issues))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_id": self.target_id,
            "profile_version": self.profile_version,
            "backend": self.backend,
            "features": copy.deepcopy(dict(self.features)),
            "resources": {key: copy.deepcopy(dict(value)) for key, value in self.resources.items()},
            "performance": copy.deepcopy(dict(self.performance)),
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetProfile":
        return cls(
            target_id=payload["target_id"],
            backend=payload["backend"],
            profile_version=payload.get("profile_version", 1),
            features=payload.get("features", {}),
            resources=payload.get("resources", {}),
            performance=payload.get("performance", {}),
            capabilities=payload.get("capabilities", ()),
            schema_version=payload.get("schema_version"),
        )

    @classmethod
    def from_json(cls, path: Path) -> "TargetProfile":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())


@dataclass(frozen=True)
class MappingCandidate:
    candidate_id: str
    compute_contract_ref: str
    target_profile_ref: str
    target_profile_fingerprint: str
    algorithm: str = "direct"
    loop_order: Tuple[str, ...] = CONV_AXES
    tiles: Mapping[str, int] = field(default_factory=lambda: {"oc": 8, "oh": 4, "ow": 16})
    vectorize_axis: Optional[str] = "ow"
    vector_length: Optional[int] = None
    tail_policy: str = "masked"
    memory_residency: Mapping[str, str] = field(
        default_factory=lambda: {"input": "l1d", "filter": "l1d", "output": "register"}
    )
    parallel_binding: Mapping[str, str] = field(default_factory=lambda: {"n": "single_thread"})
    decomposition_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "loop_order", tuple(self.loop_order))
        object.__setattr__(self, "tiles", copy.deepcopy(dict(self.tiles)))
        object.__setattr__(self, "memory_residency", copy.deepcopy(dict(self.memory_residency)))
        object.__setattr__(self, "parallel_binding", copy.deepcopy(dict(self.parallel_binding)))
        object.__setattr__(self, "decomposition_provenance", copy.deepcopy(dict(self.decomposition_provenance)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "compute_contract_ref": self.compute_contract_ref,
            "target_profile_ref": self.target_profile_ref,
            "target_profile_fingerprint": self.target_profile_fingerprint,
            "algorithm": self.algorithm,
            "loop_order": list(self.loop_order),
            "tiles": dict(self.tiles),
            "vectorize_axis": self.vectorize_axis,
            "vector_length": self.vector_length,
            "tail_policy": self.tail_policy,
            "memory_residency": dict(self.memory_residency),
            "parallel_binding": dict(self.parallel_binding),
            "decomposition_provenance": dict(self.decomposition_provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MappingCandidate":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ContractError(f"unsupported MappingCandidate schema_version: {payload.get('schema_version')!r}")
        return cls(
            candidate_id=payload["candidate_id"],
            compute_contract_ref=payload["compute_contract_ref"],
            target_profile_ref=payload["target_profile_ref"],
            target_profile_fingerprint=payload["target_profile_fingerprint"],
            algorithm=payload.get("algorithm", "direct"),
            loop_order=payload.get("loop_order", CONV_AXES),
            tiles=payload.get("tiles", {"oc": 8, "oh": 4, "ow": 16}),
            vectorize_axis=payload.get("vectorize_axis", "ow"),
            vector_length=payload.get("vector_length"),
            tail_policy=payload.get("tail_policy", "masked"),
            memory_residency=payload.get(
                "memory_residency",
                {"input": "l1d", "filter": "l1d", "output": "register"},
            ),
            parallel_binding=payload.get("parallel_binding", {"n": "single_thread"}),
            decomposition_provenance=payload.get("decomposition_provenance", {}),
        )

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.to_dict())

    def tile_working_set_by_tensor_bytes(self, contract: Conv2DContract) -> Dict[str, int]:
        tile_oc = min(self.tiles["oc"], contract.filter_shape[0])
        tile_oh = min(self.tiles["oh"], contract.output_shape[2])
        tile_ow = min(self.tiles["ow"], contract.output_shape[3])
        channels = contract.input_shape[1]
        kernel_h, kernel_w = contract.filter_shape[2:]
        input_h = (tile_oh - 1) * contract.strides[0] + kernel_h
        input_w = (tile_ow - 1) * contract.strides[1] + kernel_w
        input_bytes = channels * input_h * input_w * F32_BYTES
        filter_bytes = tile_oc * channels * kernel_h * kernel_w * F32_BYTES
        output_bytes = tile_oc * tile_oh * tile_ow * F32_BYTES
        return {"input": input_bytes, "filter": filter_bytes, "output": output_bytes}

    def tile_working_set_bytes(self, contract: Conv2DContract) -> int:
        return sum(self.tile_working_set_by_tensor_bytes(contract).values())

    def validate(self, contract: Conv2DContract, target: TargetProfile) -> None:
        issues = []
        if not self.candidate_id:
            issues.append("candidate_id is required")
        if self.compute_contract_ref != contract.fingerprint:
            issues.append("compute_contract_ref does not match the supplied contract")
        if self.target_profile_ref != target.target_ref:
            issues.append("target_profile_ref does not match the supplied target")
        if self.target_profile_fingerprint != target.fingerprint:
            issues.append("target_profile_fingerprint does not match the supplied target")
        if self.algorithm != contract.algorithm:
            issues.append("candidate algorithm does not match the compute contract")
        if set(self.loop_order) != set(CONV_AXES) or len(self.loop_order) != len(CONV_AXES):
            issues.append("loop_order must be a permutation of the Conv2D iteration axes")
        tile_values_valid = True
        for axis in ("oc", "oh", "ow"):
            value = self.tiles.get(axis)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                issues.append(f"tiles.{axis} must be a positive integer")
                tile_values_valid = False
        unknown_tiles = set(self.tiles) - {"oc", "oh", "ow"}
        if unknown_tiles:
            issues.append(f"unknown tile axes: {sorted(unknown_tiles)}")
        vector = target.features.get("vector", {})
        vectorizable_axes = set(vector.get("vectorizable_axes", ("ow",)))
        if self.vectorize_axis is not None and self.vectorize_axis not in vectorizable_axes:
            issues.append(f"vectorize_axis {self.vectorize_axis!r} is not supported by the target")
        if self.vector_length is not None and (
            isinstance(self.vector_length, bool)
            or not isinstance(self.vector_length, int)
            or self.vector_length <= 0
        ):
            issues.append("vector_length must be positive when present")
        if self.tail_policy not in {"masked", "scalar_epilogue", "padded"}:
            issues.append("tail_policy must be masked, scalar_epilogue, or padded")
        if self.tail_policy == "masked" and not vector.get("supports_masked_ops", False):
            issues.append("masked tail policy requires target masked vector operations")
        supported_dtypes = set(target.features.get("supported_dtypes", ("f32",)))
        if contract.input_dtype not in supported_dtypes:
            issues.append(f"target does not support dtype {contract.input_dtype}")
        for tensor_name, resource_id in self.memory_residency.items():
            if tensor_name not in {"input", "filter", "output"}:
                issues.append(f"unknown memory binding {tensor_name!r}")
            elif resource_id not in target.resources:
                issues.append(f"unknown target resource {resource_id!r}")
        topology = set(target.features.get("topology_dimensions", ("single_thread",)))
        for axis, dimension in self.parallel_binding.items():
            if axis not in CONV_AXES or dimension not in topology:
                issues.append(f"invalid parallel binding {axis!r} -> {dimension!r}")
        provenance_name = self.decomposition_provenance.get("name")
        provenance_source = self.decomposition_provenance.get("source_operator_ref")
        if provenance_name != self.algorithm or provenance_source != contract.fingerprint:
            issues.append("decomposition_provenance must identify the algorithm and source operator contract")
        if tile_values_valid:
            resource_usage: Dict[str, int] = {}
            tensor_bytes = self.tile_working_set_by_tensor_bytes(contract)
            for tensor_name, resource_id in self.memory_residency.items():
                if tensor_name in tensor_bytes and resource_id in target.resources:
                    resource_usage[resource_id] = resource_usage.get(resource_id, 0) + tensor_bytes[tensor_name]
            for resource_id, usage_bytes in resource_usage.items():
                capacity = target.resources[resource_id].get("capacity_bytes")
                resource_kind = target.resources[resource_id].get("kind")
                explicitly_managed = target.resources[resource_id].get(
                    "explicitly_managed",
                    resource_kind in {"register", "sram", "scratchpad", "shared_memory"},
                )
                if explicitly_managed and capacity is not None and usage_bytes > capacity:
                    issues.append(f"working set {usage_bytes} bytes exceeds {resource_id} capacity {capacity}")
        if issues:
            raise ContractError(f"invalid mapping candidate {self.candidate_id!r}: " + "; ".join(issues))
