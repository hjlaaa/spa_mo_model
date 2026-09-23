#!/usr/bin/env python3
"""Check the frozen deterministic FP32 and SPATCH BF16 GPU references.

Each execution configuration has its OWN expected. Cross-configuration checks
are reported separately. All mathematics and optimizer steps call production code.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
# Test-process execution contract, not a production config change.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from tools.validation import replay_model as b, noop_config_projection, retired_config_projection
import torch
import numpy as np

LEGACY = ROOT / "refactor_checks/v7a_gpu_fp32_frozen"
CHECKPOINTS = ("checkpoint_ot_attention", "checkpoint_encoder_fusion",
               "checkpoint_decoder_chunks", "checkpoint_graph_encoder")


def execution_environment(profile):
    result = b.environment()
    result.update({
        "precision": "FP32" if profile == "fp32" else "BF16 training autocast; FP32 refresh/final eval",
        "profile": profile,
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "supplement_is_not_production_default": True,
    })
    return result








def scalar_tensors(value):
    # Use the same atol+rtol*abs(expected) rule for Python floats and tensors.
    if isinstance(value, dict):
        return {k: scalar_tensors(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(scalar_tensors(v) for v in value)
    if isinstance(value, float):
        return torch.tensor(value, dtype=torch.float64)
    return value


def tensor_leaves(value, path=""):
    if isinstance(value, dict):
        for key, v in value.items():
            yield from tensor_leaves(v, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from tensor_leaves(v, f"{path}/{i}")
    elif isinstance(value, (torch.Tensor, np.ndarray)):
        yield path, torch.as_tensor(value)


def compare_result(actual, expected, tolerance, label):
    stats, errors = {}, []
    b.compare(scalar_tensors(actual), scalar_tensors(expected), stats, errors, label, tolerance)
    if isinstance(actual, dict) and "execution_exception" in actual:
        errors.append("Execution exception retained; matching exception text is not PASS.")
    return {"status": "FAIL" if errors else "PASS", "errors": errors, "statistics": stats,
            "exact_floating_values": not errors and all(s["max_abs"] == 0 for s in stats.values())}


def differences(actual, expected, initial, tolerance):
    """Diagnostic only: preserve first differences and near-zero Adam coordinates."""
    other = dict(tensor_leaves(expected))
    first, violations, coordinates = [], [], []
    for path, a in tensor_leaves(actual):
        e = other.get(path)
        if e is None or a.shape != e.shape or a.dtype != e.dtype or not a.is_floating_point():
            continue
        if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):
            if len(violations) < 20:
                violations.append({"path": path, "nonfinite": True})
            continue
        delta = (a.double()-e.double()).abs()
        if not delta.numel() or not bool(delta.any()):
            continue
        bad = delta > tolerance["atol"] + tolerance["rtol"]*e.double().abs()
        item = {"path": path, "dtype": str(a.dtype), "max_abs": float(delta.max()),
                "exceeds_tolerance": bool(bad.any())}
        if len(first) < 16:
            first.append(item)
        if bool(bad.any()) and len(violations) < 20:
            violations.append(item)
        if path.startswith("/parameters_after/") and bool(bad.any()):
            name = path.removeprefix("/parameters_after/")
            idx = int(delta.flatten().argmax())
            ga = actual["gradients"][name].flatten()[idx]
            ge = expected["gradients"][name].flatten()[idx]
            coordinates.append({**item, "parameter": name, "flat_index": idx,
                                "expected_grad": float(ge), "actual_grad": float(ga),
                                "initial_parameter": float(initial[name].flatten()[idx]),
                                "expected_parameter": float(e.flatten()[idx]),
                                "actual_parameter": float(a.flatten()[idx]),
                                "adam_eps": expected["optimizer_defaults"]["eps"]})
    return {"first_nonexact_tensors_in_forward_order": first,
            "first_tolerance_violations": violations, "adam_coordinates": coordinates}


def specifications(bundle):
    bf16 = bundle["profile"] == "spatch_bf16"
    base = {}
    if bf16:
        keep = ("amp_dtype", "training_loss_only", "decoder_chunk_size", "ot_attention_source_chunk_size",
                "cache_spatial_graphs", *CHECKPOINTS)
        base = {k: bundle["runner_args"][k] for k in keep}
    result = []
    for case in bundle["cases"]:
        result.append((f"{case}/main", case, "main", False, bf16, copy.deepcopy(base)))
    for enabled in (False, True):
        overrides = {**base, "training_loss_only": True, "decoder_chunk_size": 5,
                     "ot_attention_source_chunk_size": 7, "cache_spatial_graphs": True,
                     **{k: enabled for k in CHECKPOINTS}}
        result.append(("checkpoint/on" if enabled else "checkpoint/off", "three_sections",
                       "checkpoint_layout", True, enabled, overrides))
    for chunked in (False, True):
        overrides = {**base, "decoder_chunk_size": 5 if chunked else 0,
                     "ot_attention_source_chunk_size": 7 if chunked else 0,
                     **{k: False for k in CHECKPOINTS}}
        result.append(("chunk/small" if chunked else "chunk/none", "three_sections",
                       "zero_dropout", chunked, False, overrides))
    return result


def execute(bundle):
    actual, metadata = {}, {}
    helper = b.fit_runtime
    for name, case, variant, chunked, checkpoint, overrides in specifications(bundle):
        inputs = bundle["cases"][case]
        try:
            obs, meta, model, final = b.run_step(
                bundle, inputs, variant, chunked, checkpoint, fit_overrides=overrides,
                preflight=bundle["preflight"], capture_stages=True)
            actual[name] = b.snap(obs)
            metadata[name] = meta
            assert obs["train"]["autocast_enabled"] == (bundle["profile"] == "spatch_bf16")
            assert not obs["final_eval"]["autocast_enabled"]
            if name.endswith("/main"):
                init_model = b.build_model(bundle, inputs, "main")
                actual[f"{case}/sparse_init"] = b.sparse_probe(init_model, inputs, bundle["runner_args"], helper_module=helper)
                del init_model
                actual[f"{case}/refresh_helper"] = b.sparse_probe(model, inputs, bundle["runner_args"], final, helper_module=helper)
                with torch.no_grad():
                    after = b.fit_runtime.run_one_forward(
                        model, inputs["feature_dict"], inputs["spatial_loc_dict"], None, inputs["section_order"],
                        epoch=100, decoder_chunk_size=meta["fit_runtime_args"]["decoder_chunk_size"],
                        ot_attention_source_chunk_size=meta["fit_runtime_args"]["ot_attention_source_chunk_size"],
                        cache_spatial_graphs=meta["fit_runtime_args"]["cache_spatial_graphs"])
                actual[f"{case}/after_refresh_eval"] = b.snap({k: after[k] for k in b.OUTPUT_KEYS})
                del after
            del model, final
        except Exception:
            actual[f"{name}/exception"] = {"execution_exception": traceback.format_exc()}
    actual["solver/fixed_support"] = b.fixed_support_probe()
    return actual, metadata


def cross_checks(actual, bundle):
    result = {}
    for label, akey, ekey in (("checkpoint_same_chunks", "checkpoint/on", "checkpoint/off"),
                             ("chunk_dropout_zero", "chunk/small", "chunk/none")):
        if akey not in actual or ekey not in actual:
            result[label] = {"status": "FAIL", "errors": ["Missing case due to retained execution exception."]}
            continue
        a, e = actual[akey], actual[ekey]
        if label == "chunk_dropout_zero":
            a, e = b.equivalent_numeric(a), b.equivalent_numeric(e)
        result[label] = compare_result(a, e, bundle["tolerances"], label)
        result[label]["localization"] = differences(a, e, bundle["parameters"], bundle["tolerances"])
    return result


def diagnose_chunks(bundle, path):
    """Change one chunk parameter at a time; preserve original fit/math."""
    assert not path.exists() and not path.with_suffix(".actual.pt").exists()
    inputs = bundle["cases"]["three_sections"]
    none_spec = next(s for s in specifications(bundle) if s[0] == "chunk/none")
    originals, results, reports, layer_outputs = {}, {}, {}, {}
    original_builder = b.build_model
    # Decoder layer hooks expose the first BF16 rounding difference even when
    # the production loss-only branch intentionally omits reconstructions.
    def observed_builder(*a, **kw):
        model = original_builder(*a, **kw)
        active = [False]
        model.register_forward_pre_hook(lambda m, a, kw: active.__setitem__(0, m.training and torch.is_grad_enabled()), with_kwargs=True)
        model.register_forward_hook(lambda m, a, kw, out: active.__setitem__(0, False), with_kwargs=True)
        for modality, decoder in model.decoders.items():
            for index in (0, 3):
                key = f"{modality}/linear{index}"
                def observe(m, a, out, key=key):
                    if active[0]:
                        layer_outputs.setdefault(key, []).append(b.snap(out))
                decoder.network[index].register_forward_hook(observe)
        return model
    for label, decoder_size, attention_size in (("none", 0, 0), ("attention_only", 0, 7), ("decoder_only", 5, 0)):
        layer_outputs.clear()
        overrides = {**none_spec[-1], "decoder_chunk_size": decoder_size,
                     "ot_attention_source_chunk_size": attention_size}
        with patch.object(b, "build_model", observed_builder):
            obs, meta, _, _ = b.run_step(bundle, inputs, "zero_dropout", False, False,
                                       fit_overrides=overrides, preflight=bundle["preflight"], capture_stages=True)
        actual = b.equivalent_numeric(obs)
        results[label] = actual
        originals[label] = {k: torch.cat(parts, dim=0) for k, parts in layer_outputs.items()}
        reference = bundle["expected"]["chunk/none"] if label == "none" else results["none"]
        reference = b.equivalent_numeric(reference)
        reports[label] = compare_result(actual, reference, bundle["tolerances"], label)
        reports[label]["localization"] = differences(actual, reference, bundle["parameters"], bundle["tolerances"])
        reports[label]["fit_runtime_args"] = meta["fit_runtime_args"]
        if label != "none":
            reports[label]["decoder_layers"] = compare_result(originals[label], originals["none"], bundle["tolerances"], "decoder_layers")
            reports[label]["decoder_first_differences"] = differences(originals[label], originals["none"], {}, bundle["tolerances"])
    torch.save({"observations": results, "decoder_layers": originals}, path.with_suffix(".actual.pt"))
    report = {"profile": bundle["profile"], "status": "DIAGNOSTIC_FAILURES_RETAINED",
              "comparisons": reports, "environment": execution_environment(bundle["profile"]),
              "supplement_script_sha256": b.sha(__file__),
              "scope": "none repeats frozen expected; others change one chunk size only; no expected modified; raw decoder outputs concatenate in original section/spot order."}
    b.json_write(path, report)
    print(json.dumps({k: {"status": v["status"], "statistics": v["statistics"]} for k, v in reports.items()}, indent=2))



# P2 schema-only adaptation: remove the retired always-None dense placeholder
# from loaded historical inputs and expected values in memory, never on disk.
# This is the sole permitted schema projection. No tensor or other key changes.
P2_RETIRED_PLACEHOLDER_PATHS = []
def p2_project_frozen_dense_placeholder(value, path="bundle"):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key == "P_dense":
                assert child is None, f"Nonempty dense expected cannot be projected: {child_path}"
                P2_RETIRED_PLACEHOLDER_PATHS.append(child_path)
            else:
                result[key] = p2_project_frozen_dense_placeholder(child, child_path)
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(p2_project_frozen_dense_placeholder(child, f"{path}/{i}")
                           for i, child in enumerate(value))
    return value



def p2_assert_projection_identity(original, projected, path="bundle"):
    if isinstance(original, dict):
        assert type(projected) is type(original), path
        assert list(projected) == [key for key in original if key != "P_dense"], path
        for key in projected:
            p2_assert_projection_identity(original[key], projected[key], f"{path}/{key}")
    elif isinstance(original, (list, tuple)):
        assert type(original) is type(projected) and len(original) == len(projected), path
        for index in range(len(original)):
            p2_assert_projection_identity(original[index], projected[index], f"{path}/{index}")
    else:
        # All tensors, parameter/state values, RNG, indices and scalar leaves are
        # retained by object identity; this projection cannot alter their values.
        assert projected is original, f"Non-placeholder leaf changed: {path}"


def p2_require_no_dense_placeholder(value, path="actual"):
    if isinstance(value, dict):
        assert "P_dense" not in value, f"Retired placeholder in actual: {path}"
        for key, child in value.items():
            p2_require_no_dense_placeholder(child, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            p2_require_no_dense_placeholder(child, f"{path}/{i}")


# D4 schema-only input projection, restricted to two retired config choices.
# Expected observations, tensor/state/RNG leaves and all active config survive by identity.
D4_RETIRED_CONFIG_PATHS = []
def d4_project_frozen_config(bundle):
    result = dict(bundle)
    result["config"] = dict(bundle["config"])
    result["config"]["uot"] = dict(bundle["config"]["uot"])
    retired = {"dynamic_refresh_source": "ot", "update_from_final_embedding": False}
    for key, expected in retired.items():
        assert type(bundle["config"]["uot"][key]) is type(expected)
        assert bundle["config"]["uot"][key] == expected
        del result["config"]["uot"][key]
        D4_RETIRED_CONFIG_PATHS.append(f"bundle/config/uot/{key}")
    def same(original, projected, path="bundle"):
        if isinstance(original, dict):
            omitted = set(retired) if path == "bundle/config/uot" else set()
            assert list(projected) == [key for key in original if key not in omitted], path
            for key in projected:
                same(original[key], projected[key], f"{path}/{key}")
        elif isinstance(original, (list, tuple)):
            assert type(projected) is type(original) and len(projected) == len(original), path
            for i in range(len(original)):
                same(original[i], projected[i], f"{path}/{i}")
        else:
            assert projected is original, f"Nonretired config/expected leaf changed: {path}"
    same(bundle, result)
    assert result["expected"] is bundle["expected"]
    return result


# D3 removes only three inactive context-gate config choices and the empty
# gate-only output mapping. No tensor, numeric leaf, RNG, parameter or real
# spatial refresh context is projected; old values are asserted before removal.
D3_RETIRED_CONFIG_PATHS = []
D3_RETIRED_EMPTY_OUTPUT_PATHS = []
def d3_project_frozen_gate(bundle):
    retired = {"context_gate_enabled": False, "context_consistency_backprop_to_alpha": False, "context_eps": 1e-8}
    def project(value, path="bundle"):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                child_path = f"{path}/{key}"
                if path == "bundle/config/ot_attention" and key in retired:
                    expected = retired[key]
                    assert type(child) is type(expected) and child == expected, child_path
                    D3_RETIRED_CONFIG_PATHS.append(child_path)
                elif key == "context_embeddings":
                    assert isinstance(child, dict) and not child, child_path
                    assert path.startswith("bundle/expected/"), child_path
                    D3_RETIRED_EMPTY_OUTPUT_PATHS.append(child_path)
                else:
                    result[key] = project(child, child_path)
            return result
        if isinstance(value, (tuple, list)):
            return type(value)(project(child, f"{path}/{i}") for i, child in enumerate(value))
        return value
    result = project(bundle)
    assert len(D3_RETIRED_CONFIG_PATHS) == 3
    assert D3_RETIRED_EMPTY_OUTPUT_PATHS
    omitted = set(D3_RETIRED_CONFIG_PATHS + D3_RETIRED_EMPTY_OUTPUT_PATHS)
    def identical(original, projected, path="bundle"):
        if isinstance(original, dict):
            assert type(projected) is type(original)
            assert list(projected) == [k for k in original if f"{path}/{k}" not in omitted], path
            for key in projected:
                identical(original[key], projected[key], f"{path}/{key}")
        elif isinstance(original, (tuple, list)):
            assert type(original) is type(projected) and len(original) == len(projected), path
            for i in range(len(original)):
                identical(original[i], projected[i], f"{path}/{i}")
        else:
            assert projected is original, f"Nonretired leaf changed: {path}"
    identical(bundle, result)
    return result


# D2 approved schema retirement. Restrict projection to the two unused learned
# self weights by full name, their exact None-gradient observations, and the
# single retired self-path config. No effective weight, numeric observation, RNG,
# graph, prior or optimizer value is altered. Original fixture stays read-only.
D2_RETIRED_PARAMETER_NAMES = (
    "graphsage.self_linear.weight", "post_ot_graphsage.self_linear.weight",
)
D2_RETIRED_PATHS = []
def d2_project_frozen_self_path(bundle):
    names = set(D2_RETIRED_PARAMETER_NAMES)
    paths = {"bundle/config/graphsage/self_path_mode"}
    assert bundle["config"]["graphsage"]["self_path_mode"] == "no_self_linear"
    for name in names:
        weight = bundle["parameters"][name]
        schema = bundle["parameter_schema"][name]
        assert isinstance(weight, torch.Tensor) and list(weight.shape) == [128, 128]
        assert weight.dtype == torch.float32
        assert schema == {"shape": [128, 128], "dtype": "torch.float32", "requires_grad": True}
        paths.add(f"bundle/parameters/{name}")
        paths.add(f"bundle/parameter_schema/{name}")
    step_names = {"two_sections/main", "three_sections/main", "checkpoint/off",
                  "checkpoint/on", "chunk/none", "chunk/small"}
    observed_steps = {case for case, observation in bundle["expected"].items()
                      if "gradients" in observation}
    assert observed_steps == step_names
    for case in step_names:
        observation = bundle["expected"][case]
        for name in names:
            assert observation["gradients"][name] is None, (case, name)
            assert name not in observation["parameters_after"], (case, name)
            paths.add(f"bundle/expected/{case}/gradients/{name}")
    def project(value, path="bundle"):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                child_path = f"{path}/{key}"
                if child_path in paths:
                    D2_RETIRED_PATHS.append(child_path)
                else:
                    result[key] = project(child, child_path)
            return result
        if isinstance(value, (tuple, list)):
            return type(value)(project(child, f"{path}/{i}") for i, child in enumerate(value))
        return value
    result = project(bundle)
    assert set(D2_RETIRED_PATHS) == paths and len(D2_RETIRED_PATHS) == 17
    def identical(original, projected, path="bundle"):
        if isinstance(original, dict):
            assert type(projected) is type(original), path
            assert list(projected) == [key for key in original if f"{path}/{key}" not in paths], path
            for key in projected:
                identical(original[key], projected[key], f"{path}/{key}")
        elif isinstance(original, (tuple, list)):
            assert type(projected) is type(original) and len(projected) == len(original), path
            for i in range(len(original)):
                identical(original[i], projected[i], f"{path}/{i}")
        else:
            assert projected is original, f"Unapproved leaf change: {path}"
    identical(bundle, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "diagnose-chunks"))
    parser.add_argument("--profile", choices=("fp32", "spatch_bf16"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    directory = (ROOT / args.baseline).resolve()
    assert directory.is_relative_to((ROOT / "refactor_checks").resolve())
    assert torch.cuda.is_available(), "GPU required; no fallback"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    b.faiss.omp_set_num_threads(1)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.time()
    if args.operation == "diagnose-chunks":
        assert args.report
        path = (ROOT / args.report).resolve()
        assert path.is_relative_to((ROOT / "refactor_checks").resolve())
        manifest = json.loads((directory / "baseline.json").read_text())
        assert b.sha(directory / "fixture.pt") == manifest["fixture_sha256"]
        bundle = torch.load(directory / "fixture.pt", map_location="cpu", weights_only=False)
        assert bundle["profile"] == args.profile and execution_environment(args.profile) == bundle["environment"]
        diagnose_chunks(bundle, path)
        return
    assert args.report, "check requires fresh report path"
    path = (ROOT / args.report).resolve()
    assert path.is_relative_to((ROOT / "refactor_checks").resolve())
    assert not path.exists() and not path.with_suffix(".actual.pt").exists()
    manifest = json.loads((directory / "baseline.json").read_text())
    assert b.sha(directory / "fixture.pt") == manifest["fixture_sha256"]
    bundle = torch.load(directory / "fixture.pt", map_location="cpu", weights_only=False)
    assert bundle["profile"] == args.profile
    assert execution_environment(args.profile) == bundle["environment"]
    frozen_bundle = bundle
    bundle = p2_project_frozen_dense_placeholder(frozen_bundle)
    p2_assert_projection_identity(frozen_bundle, bundle)
    assert len(P2_RETIRED_PLACEHOLDER_PATHS) == 40
    bundle = d4_project_frozen_config(bundle)
    bundle = d3_project_frozen_gate(bundle)
    bundle = d2_project_frozen_self_path(bundle)
    from tools.validation.noop_config_projection import project_bundle
    bundle = project_bundle(bundle)
    from tools.validation.retired_config_projection import project_p2_config_bundle
    bundle = project_p2_config_bundle(bundle)
    actual, metadata = execute(bundle)
    p2_require_no_dense_placeholder(actual)
    own = {k: compare_result(actual[k], v, bundle["tolerances"], k)
           if k in actual else {"status": "FAIL", "errors": ["Expected case missing"]}
           for k, v in bundle["expected"].items()}
    if set(actual) != set(bundle["expected"]):
        own["case_schema"] = {"status": "FAIL", "errors": ["Actual/expected case keys differ"]}
    # Only actual output, never consumed as expected by check.
    torch.save(actual, path.with_suffix(".actual.pt"))
    cross = cross_checks(actual, bundle)
    failures = [k for k, v in own.items() if v["status"] == "FAIL"]
    cross_failures = [k for k, v in cross.items() if v["status"] == "FAIL"]
    result = {
        "p2_retired_input_config_projection": {"paths": [".".join(path) for path in retired_config_projection.P2_RETIRED_DEFAULTS], "old_default_values_exactly_asserted": True, "all_other_leaves_object_identical": True},
        "b8_noop_config_projection": {"paths": [".".join(path) for path in noop_config_projection.NOOP_DEFAULTS], "old_default_values_exactly_asserted": True, "all_other_leaves_object_identical": True, "no_new_parameter_schema_projection": True},
        "d2_approved_schema_retirement": {"parameter_names": D2_RETIRED_PARAMETER_NAMES, "projected_paths": D2_RETIRED_PATHS, "retired_gradients_required_none": True, "all_other_leaves_object_identical": True, "strict_named_state_shape_dtype_requires_grad_check": True, "rng_restored_from_original_fixture_before_preflight": True, "same_seed_construction_equivalence_not_claimed": True},
        "d3_gate_adaptation": {"projected_config_paths": D3_RETIRED_CONFIG_PATHS, "projected_empty_gate_output_paths": D3_RETIRED_EMPTY_OUTPUT_PATHS, "all_other_leaves_object_identical": True, "no_tensor_or_real_refresh_context_removed": True},
        "d4_config_adaptation": {"projected_paths": D4_RETIRED_CONFIG_PATHS, "required_frozen_values": {"dynamic_refresh_source": "ot", "update_from_final_embedding": False}, "all_other_leaf_objects_identical": True, "expected_observations_identical": True},
        "p2_schema_adaptation": {"removed_field": "P_dense", "required_old_value": None, "projected_frozen_paths": P2_RETIRED_PLACEHOLDER_PATHS, "actual_has_retired_placeholder": False, "all_other_fields_and_comparisons_unchanged": True, "all_noncontainer_leaf_objects_identical_to_frozen": True},
        "operation": args.operation, "profile": args.profile,
        "status": "FAIL" if failures or cross_failures else "PASS_SCOPED",
        "same_configuration_failures": failures, "cross_configuration_failures": cross_failures,
        "same_configuration": own, "cross_configuration": cross, "execution": metadata,
        "environment": execution_environment(args.profile), "current_provenance": b.provenance(),
        "supplement_script_sha256": b.sha(__file__), "elapsed_seconds": time.time()-started,
        "refresh_original_loops": b.static_refresh_audit(),
        "coverage_limits": [
            "Deterministic verification settings differ from production defaults; old FAILs retained.",
            "Real one-epoch fit and FP32 final eval; scheduler and refresh helpers dynamic, epoch100+ whole loops static.",
            "Synthetic inputs; no SPATCH preprocessing/cache arrays, production output saver, analysis or real data.",
            "Small boundary chunks separate from actual configured SPATCH chunk sizes.",
            "Existing R1/B4 are not repaired; no NaN equality or tolerance change.",
        ],
    }
    b.json_write(path, result)
    print(json.dumps({k: result[k] for k in ("status", "profile", "same_configuration_failures",
                                           "cross_configuration_failures", "elapsed_seconds")}, indent=2))
    print(str(path))
    if failures or cross_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
