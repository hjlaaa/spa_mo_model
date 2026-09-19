#!/usr/bin/env python3
"""Inspect fixed GPU results; an unexplained exit 1 is never accepted."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check(profile, process_exit_code, report_name):
    name = "fp32" if profile == "fp32" else "bf16"
    baseline = "fp32_deterministic" if profile == "fp32" else "spatch_bf16_deterministic"
    report = json.loads((ROOT / report_name).read_text())
    historical = json.loads((ROOT / "refactor_checks/p0_supplement" / baseline / "replay_report.json").read_text())
    assert process_exit_code == 1, process_exit_code
    assert report["same_configuration_failures"] == []
    assert len(report["same_configuration"]) == 13
    for key, result in report["same_configuration"].items():
        assert result["status"] == "PASS" and result["exact_floating_values"], (key, result)
        assert not result["errors"], (key, result)
        for statistics in result["statistics"].values():
            assert statistics["max_abs"] == statistics["max_rel"] == 0, (key, statistics)
    assert report["cross_configuration_failures"] == ["chunk_dropout_zero"]
    assert report["cross_configuration"] == historical["cross_configuration"]
    checkpoint = report["cross_configuration"]["checkpoint_same_chunks"]
    assert checkpoint["status"] == "PASS" and checkpoint["exact_floating_values"]
    retired_input = report["p2_retired_input_config_projection"]
    assert len(retired_input["paths"]) == 6 and retired_input["old_default_values_exactly_asserted"]
    assert retired_input["all_other_leaves_object_identical"]
    b8 = report["b8_noop_config_projection"]
    assert len(b8["paths"]) == 23 and b8["old_default_values_exactly_asserted"]
    assert b8["all_other_leaves_object_identical"] and b8["no_new_parameter_schema_projection"]
    adaptation = report["d2_approved_schema_retirement"]
    assert len(adaptation["projected_paths"]) == 17
    assert adaptation["retired_gradients_required_none"]
    assert adaptation["all_other_leaves_object_identical"]
    assert adaptation["strict_named_state_shape_dtype_requires_grad_check"]
    return {
        "fixed_gate": "PASS / APPROVED_SCHEMA_RETIREMENT",
        "B8_input_NOOP_config_projection": b8,
        "P2_retired_input_config_projection": retired_input,
        "same_configuration_cases": 13,
        "max_abs": 0,
        "max_rel": 0,
        "checkpoint_same_chunks": "PASS exact zero",
        "cross_configuration_complete_object_identical_to_frozen_replay": True,
        "cross_chunk_status": "FAIL / KNOWN_NUMERICAL_SENSITIVITY",
        "cross_chunk_parameters_max_abs": report["cross_configuration"]["chunk_dropout_zero"]["statistics"]["parameters_after"]["max_abs"],
        "process_exit_code": process_exit_code,
        "report": report_name,
        "D2_retired_parameters": adaptation["parameter_names"],
        "same_seed_initialization_equivalence_claimed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fp32", "spatch_bf16"), required=True)
    parser.add_argument("--process-exit-code", type=int, required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.profile, args.process_exit_code, args.report), indent=2))
