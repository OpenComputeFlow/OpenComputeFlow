from __future__ import annotations

from pathlib import Path

from opencomputeflow import (
    CompilationTrace,
    Conv2DContract,
    DecisionRecord,
    MappingCandidate,
    Measurement,
    RejectedCandidate,
    TargetProfile,
    estimate_direct_conv,
)


def build_phase0_fixture(repo_root: Path):
    target = TargetProfile.from_json(repo_root / "profiles" / "rvv-example.json")
    contract = Conv2DContract(input_shape=(1, 3, 8, 16), filter_shape=(8, 3, 3, 3))
    candidate = MappingCandidate(
        candidate_id="conv_ow16",
        compute_contract_ref=contract.fingerprint,
        target_profile_ref=target.target_ref,
        target_profile_fingerprint=target.fingerprint,
        decomposition_provenance={
            "name": "direct",
            "source_operator_ref": contract.fingerprint,
        },
    )
    estimate = estimate_direct_conv(contract, candidate, target)
    measurement = Measurement(
        measurement_id="deterministic-cycle-model-run-1",
        candidate_ref=candidate.fingerprint,
        target_profile_fingerprint=target.fingerprint,
        source_kind="cycle_accurate_model",
        metric="latency_cycles",
        unit="cycles",
        collected_at="2026-08-03T00:00:00+08:00",
        samples=(4600.0, 4550.0, 4580.0, 4570.0, 4560.0),
        warmup_runs=10,
        environment={
            "device_id": "deterministic-test-cycle-model",
            "backend_version": "rvv-adapter-dev",
            "clock_policy": "modeled-fixed-clock",
            "threads": 1,
        },
    )
    rejected_ref = "a" * 64
    decision = DecisionRecord(
        site="conv2d_0",
        candidates_generated=(candidate.fingerprint, rejected_ref),
        candidates_legal=(candidate.fingerprint,),
        selected_ref=candidate.fingerprint,
        estimate_refs={candidate.fingerprint: estimate.fingerprint},
        rejected=(
            RejectedCandidate(
                candidate_ref=rejected_ref,
                reason_code="unsupported_vector_axis",
                detail="target profile does not allow vectorization of kh",
            ),
        ),
    )
    trace = CompilationTrace(
        trace_id="phase0-example-trace",
        input_fingerprint=contract.fingerprint,
        target_profile_ref=target.target_ref,
        target_profile_fingerprint=target.fingerprint,
        backend_version="rvv-adapter-dev",
        cost_model_id=estimate.model_id,
        calibration_id=None,
        decisions=(decision,),
        measurement_refs=(measurement.fingerprint,),
    )
    return {
        "contract": contract,
        "target": target,
        "candidate": candidate,
        "estimate": estimate,
        "measurement": measurement,
        "decision": decision,
        "trace": trace,
    }
