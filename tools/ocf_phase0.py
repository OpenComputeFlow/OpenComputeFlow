#!/usr/bin/env python3
"""Print the Phase 0 Conv2D contract, mapping candidate, and estimate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from opencomputeflow import Conv2DContract, MappingCandidate, TargetProfile, estimate_direct_conv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "profiles" / "rvv-example.json",
        help="versioned Target Profile JSON",
    )
    args = parser.parse_args()

    target = TargetProfile.from_json(args.profile)
    contract = Conv2DContract(
        input_shape=(1, 3, 8, 16),
        filter_shape=(8, 3, 3, 3),
    )
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
    print(json.dumps({
        "contract": contract.to_dict(),
        "contract_fingerprint": contract.fingerprint,
        "target": target.to_dict(),
        "candidate": candidate.to_dict(),
        "estimate": estimate.to_dict(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
