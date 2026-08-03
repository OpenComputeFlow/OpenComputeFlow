#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

from phase0_fixture import build_phase0_fixture


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = ("contract", "target", "candidate", "estimate", "measurement", "trace")


def write_bundle(path: Path, bundle) -> None:
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_checker(checker: Path, bundle: Path, golden: Path, expect_success: bool) -> None:
    result = subprocess.run(
        [str(checker), str(bundle), str(golden)],
        text=True,
        capture_output=True,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(f"checker rejected valid bundle:\n{result.stdout}\n{result.stderr}")
    if not expect_success and result.returncode == 0:
        raise AssertionError("checker accepted an invalid bundle")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    args = parser.parse_args()

    fixture = build_phase0_fixture(REPO_ROOT)
    bundle = {name: fixture[name].to_dict() for name in ARTIFACT_NAMES}
    with tempfile.TemporaryDirectory(prefix="ocf-contract-reader-") as temp_dir:
        temp_path = Path(temp_dir)
        valid_path = temp_path / "valid.json"
        write_bundle(valid_path, bundle)
        run_checker(args.checker, valid_path, args.golden, expect_success=True)

        stale_profile = copy.deepcopy(bundle)
        stale_profile["candidate"]["target_profile_fingerprint"] = "0" * 64
        stale_path = temp_path / "stale-profile.json"
        write_bundle(stale_path, stale_profile)
        run_checker(args.checker, stale_path, args.golden, expect_success=False)

        emulator_measurement = copy.deepcopy(bundle)
        emulator_measurement["measurement"]["source_kind"] = "functional_emulator"
        emulator_path = temp_path / "emulator-measurement.json"
        write_bundle(emulator_path, emulator_measurement)
        run_checker(args.checker, emulator_path, args.golden, expect_success=False)

        unsupported_version = copy.deepcopy(bundle)
        unsupported_version["contract"]["schema_version"] = 2
        version_path = temp_path / "unsupported-version.json"
        write_bundle(version_path, unsupported_version)
        run_checker(args.checker, version_path, args.golden, expect_success=False)

        wrong_unit = copy.deepcopy(bundle)
        wrong_unit["measurement"]["unit"] = "us"
        unit_path = temp_path / "wrong-unit.json"
        write_bundle(unit_path, wrong_unit)
        run_checker(args.checker, unit_path, args.golden, expect_success=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
