from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from opencomputeflow import (  # noqa: E402
    ContractError,
    Conv2DContract,
    MappingCandidate,
    PerformanceEstimate,
    TargetProfile,
    direct_conv2d,
    estimate_direct_conv,
)


PROFILE_PATH = REPO_ROOT / "profiles" / "rvv-example.json"


class Conv2DContractTest(unittest.TestCase):
    def test_output_shape_and_work_summary(self) -> None:
        contract = Conv2DContract(
            input_shape=(1, 3, 8, 8),
            filter_shape=(4, 3, 3, 3),
            strides=(2, 2),
            padding=(1, 1, 1, 1),
        )
        self.assertEqual(contract.output_shape, (1, 4, 4, 4))
        summary = contract.work_summary()
        self.assertEqual(summary["multiply_accumulates"], 1 * 4 * 4 * 4 * 3 * 3 * 3)
        self.assertEqual(summary["flops"], 2 * summary["multiply_accumulates"])
        self.assertEqual(contract, Conv2DContract.from_dict(contract.to_dict()))
        self.assertEqual(contract.fingerprint, Conv2DContract.from_dict(contract.to_dict()).fingerprint)
        self.assertTrue(contract.to_dict()["numerical_policy"]["allow_fma_contraction"])

    def test_design_mvp_shapes(self) -> None:
        conv_3x3 = Conv2DContract(
            input_shape=(1, 3, 8, 8),
            filter_shape=(8, 3, 3, 3),
            padding=(1, 1, 1, 1),
        )
        conv_7x7 = Conv2DContract(
            input_shape=(1, 3, 224, 224),
            filter_shape=(64, 3, 7, 7),
            strides=(2, 2),
            padding=(3, 3, 3, 3),
        )
        self.assertEqual(conv_3x3.output_shape, (1, 8, 8, 8))
        self.assertEqual(conv_7x7.output_shape, (1, 64, 112, 112))

    def test_rejects_invalid_mvp_contract(self) -> None:
        with self.assertRaisesRegex(ContractError, "filter input channels"):
            Conv2DContract(input_shape=(1, 3, 4, 4), filter_shape=(2, 2, 3, 3))
        with self.assertRaisesRegex(ContractError, "groups must be 1"):
            Conv2DContract(input_shape=(1, 3, 4, 4), filter_shape=(2, 3, 3, 3), groups=2)
        with self.assertRaisesRegex(ContractError, "output spatial dimensions"):
            Conv2DContract(input_shape=(1, 1, 2, 2), filter_shape=(1, 1, 5, 5))
        with self.assertRaisesRegex(ContractError, "supports f32"):
            Conv2DContract(input_shape=(1, 1, 2, 2), filter_shape=(1, 1, 1, 1), input_dtype="f16")
        with self.assertRaisesRegex(ContractError, "dilation must be"):
            Conv2DContract(input_shape=(1, 1, 4, 4), filter_shape=(1, 1, 1, 1), dilation=(2, 2))
        with self.assertRaisesRegex(ContractError, "layouts must be"):
            Conv2DContract(input_shape=(1, 1, 2, 2), filter_shape=(1, 1, 1, 1), input_layout="NHWC")
        with self.assertRaisesRegex(ContractError, "declared output_shape"):
            payload = Conv2DContract(
                input_shape=(1, 1, 3, 3),
                filter_shape=(1, 1, 1, 1),
            ).to_dict()
            payload["output_shape"] = [1, 1, 2, 2]
            Conv2DContract.from_dict(payload)


class ReferenceConv2DTest(unittest.TestCase):
    def test_cross_correlation_without_padding(self) -> None:
        contract = Conv2DContract(input_shape=(1, 1, 3, 3), filter_shape=(1, 1, 2, 2))
        result = direct_conv2d(contract, list(range(1, 10)), [1, 1, 1, 1])
        self.assertEqual(result, (12.0, 16.0, 24.0, 28.0))

    def test_padding_uses_zero_values(self) -> None:
        contract = Conv2DContract(
            input_shape=(1, 1, 1, 1),
            filter_shape=(1, 1, 3, 3),
            padding=(1, 1, 1, 1),
        )
        self.assertEqual(direct_conv2d(contract, [2.0], [1.0] * 9), (2.0,))

    def test_rejects_wrong_buffer_length(self) -> None:
        contract = Conv2DContract(input_shape=(1, 1, 2, 2), filter_shape=(1, 1, 1, 1))
        with self.assertRaisesRegex(ContractError, "input has 3 values"):
            direct_conv2d(contract, [1.0, 2.0, 3.0], [1.0])


class MappingAndPerformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.target = TargetProfile.from_json(PROFILE_PATH)
        cls.contract = Conv2DContract(input_shape=(1, 3, 8, 16), filter_shape=(8, 3, 3, 3))

    def candidate(self, **changes):
        values = {
            "candidate_id": "conv_ow16",
            "compute_contract_ref": self.contract.fingerprint,
            "target_profile_ref": self.target.target_ref,
            "target_profile_fingerprint": self.target.fingerprint,
            "decomposition_provenance": {
                "name": "direct",
                "source_operator_ref": self.contract.fingerprint,
            },
        }
        values.update(changes)
        return MappingCandidate(**values)

    def test_profile_is_versioned_and_round_trips(self) -> None:
        payload = self.target.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(self.target, TargetProfile.from_dict(json.loads(json.dumps(payload))))

    def test_valid_candidate_and_capacity(self) -> None:
        candidate = self.candidate()
        candidate.validate(self.contract, self.target)
        self.assertLess(candidate.tile_working_set_bytes(self.contract), self.target.resources["l1d"]["capacity_bytes"])
        restored = MappingCandidate.from_dict(json.loads(json.dumps(candidate.to_dict())))
        self.assertEqual(candidate, restored)
        self.assertEqual(candidate.fingerprint, restored.fingerprint)

    def test_rejects_unsupported_mapping(self) -> None:
        with self.assertRaisesRegex(ContractError, "vectorize_axis"):
            self.candidate(vectorize_axis="kh").validate(self.contract, self.target)
        profile_payload = self.target.to_dict()
        profile_payload["resources"]["l1d"]["capacity_bytes"] = 1024
        small_cache_target = TargetProfile.from_dict(profile_payload)
        self.candidate(target_profile_fingerprint=small_cache_target.fingerprint).validate(
            self.contract,
            small_cache_target,
        )
        profile_payload["resources"]["l1d"].update({"kind": "scratchpad", "explicitly_managed": True})
        small_explicit_target = TargetProfile.from_dict(profile_payload)
        with self.assertRaisesRegex(ContractError, "working set"):
            self.candidate(target_profile_fingerprint=small_explicit_target.fingerprint).validate(
                self.contract,
                small_explicit_target,
            )
        with self.assertRaisesRegex(ContractError, "decomposition_provenance"):
            self.candidate(decomposition_provenance={}).validate(self.contract, self.target)
        with self.assertRaisesRegex(ContractError, "target_profile_fingerprint"):
            self.candidate(target_profile_fingerprint="stale").validate(self.contract, self.target)

    def test_estimate_is_explainable_and_deterministic(self) -> None:
        candidate = self.candidate()
        first = estimate_direct_conv(self.contract, candidate, self.target)
        second = estimate_direct_conv(self.contract, candidate, self.target)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first, PerformanceEstimate.from_dict(json.loads(json.dumps(first.to_dict()))))
        self.assertGreater(first.flops, 0)
        self.assertGreater(first.latency_cycles, 0.0)
        self.assertIn(first.predicted_bottleneck, {"compute", "memory", "balanced"})
        self.assertEqual(set(first.traffic_bytes), {"input", "filter", "output"})
        self.assertEqual(first.to_dict()["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
