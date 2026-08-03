from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

try:
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError
except ImportError:
    Draft202012Validator = None
    FormatChecker = None
    ValidationError = Exception

from opencomputeflow import (  # noqa: E402
    CompilationTrace,
    ContractError,
    DecisionRecord,
    Measurement,
)
from phase0_fixture import build_phase0_fixture  # noqa: E402


SCHEMA_PATH = REPO_ROOT / "schemas" / "opencomputeflow-v1.schema.json"
PROFILE_PATH = REPO_ROOT / "profiles" / "rvv-example.json"
GOLDEN_FINGERPRINTS_PATH = REPO_ROOT / "tests" / "golden" / "phase0-fingerprints-v1.json"


class EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_phase0_fixture(REPO_ROOT)
        for name, value in fixture.items():
            setattr(cls, name, value)

    def test_measurement_summary_and_round_trip(self) -> None:
        self.assertEqual(self.measurement.summary, {"min": 4550.0, "median": 4570.0, "p90": 4600.0, "max": 4600.0})
        restored = Measurement.from_dict(json.loads(json.dumps(self.measurement.to_dict())))
        self.assertEqual(restored, self.measurement)
        self.assertEqual(restored.fingerprint, self.measurement.fingerprint)

    def test_measurement_rejects_unqualified_source_and_tampered_summary(self) -> None:
        with self.assertRaisesRegex(ContractError, "source_kind"):
            Measurement(
                measurement_id="bad",
                candidate_ref=self.candidate.fingerprint,
                target_profile_fingerprint=self.target.fingerprint,
                source_kind="functional_emulator",
                metric="latency_cycles",
                unit="cycles",
                collected_at="2026-08-02T00:00:00+08:00",
                samples=(1.0,),
                warmup_runs=0,
                environment={
                    "device_id": "emulator",
                    "backend_version": "dev",
                    "clock_policy": "host",
                    "threads": 1,
                },
            )
        payload = self.measurement.to_dict()
        payload["summary"]["median"] = 1.0
        with self.assertRaisesRegex(ContractError, "summary does not match"):
            Measurement.from_dict(payload)
        payload = self.measurement.to_dict()
        payload["unit"] = "us"
        with self.assertRaisesRegex(ContractError, "requires unit"):
            Measurement.from_dict(payload)
        payload = self.measurement.to_dict()
        payload["collected_at"] = "2026-08-02T00:00:00"
        with self.assertRaisesRegex(ContractError, "timezone"):
            Measurement.from_dict(payload)

    def test_decision_and_trace_round_trip(self) -> None:
        restored = CompilationTrace.from_dict(json.loads(json.dumps(self.trace.to_dict())))
        self.assertEqual(restored, self.trace)
        self.assertEqual(restored.fingerprint, self.trace.fingerprint)
        with self.assertRaisesRegex(ContractError, "selected_ref"):
            DecisionRecord(
                site="conv2d_0",
                candidates_generated=(self.candidate.fingerprint,),
                candidates_legal=(self.candidate.fingerprint,),
                selected_ref="b" * 64,
                estimate_refs={self.candidate.fingerprint: self.estimate.fingerprint},
            )

    def test_golden_fingerprints(self) -> None:
        expected = json.loads(GOLDEN_FINGERPRINTS_PATH.read_text(encoding="utf-8"))
        actual = {
            "contract": self.contract.fingerprint,
            "target": self.target.fingerprint,
            "candidate": self.candidate.fingerprint,
            "estimate": self.estimate.fingerprint,
            "measurement": self.measurement.fingerprint,
            "trace": self.trace.fingerprint,
        }
        self.assertEqual(actual, expected)


@unittest.skipUnless(Draft202012Validator is not None, "install requirements-dev.txt for schema tests")
class JsonSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        EvidenceTest.setUpClass()
        for name in ("target", "contract", "candidate", "estimate", "measurement", "decision", "trace"):
            setattr(cls, name, getattr(EvidenceTest, name))
        cls.schema_bundle = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema_bundle)

    def validate_definition(self, definition: str, instance) -> None:
        schema = {
            "$schema": self.schema_bundle["$schema"],
            "$defs": self.schema_bundle["$defs"],
            "$ref": f"#/$defs/{definition}",
        }
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)

    def test_all_phase0_artifacts_match_schema(self) -> None:
        artifacts = {
            "conv2dContract": self.contract.to_dict(),
            "targetProfile": self.target.to_dict(),
            "mappingCandidate": self.candidate.to_dict(),
            "performanceEstimate": self.estimate.to_dict(),
            "measurement": self.measurement.to_dict(),
            "compilationTrace": self.trace.to_dict(),
        }
        for definition, artifact in artifacts.items():
            with self.subTest(definition=definition):
                self.validate_definition(definition, artifact)
                Draft202012Validator(
                    self.schema_bundle,
                    format_checker=FormatChecker(),
                ).validate(artifact)

    def test_schema_rejects_field_mixing_and_invalid_ranges(self) -> None:
        estimate = copy.deepcopy(self.estimate.to_dict())
        estimate["samples"] = [4550.0]
        with self.assertRaises(ValidationError):
            self.validate_definition("performanceEstimate", estimate)

        measurement = copy.deepcopy(self.measurement.to_dict())
        measurement["source_kind"] = "functional_emulator"
        with self.assertRaises(ValidationError):
            self.validate_definition("measurement", measurement)

        measurement = copy.deepcopy(self.measurement.to_dict())
        measurement["unit"] = "us"
        with self.assertRaises(ValidationError):
            self.validate_definition("measurement", measurement)

        target = copy.deepcopy(self.target.to_dict())
        target["performance"]["model_confidence"] = 1.1
        with self.assertRaises(ValidationError):
            self.validate_definition("targetProfile", target)


if __name__ == "__main__":
    unittest.main()
