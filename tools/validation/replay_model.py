#!/usr/bin/env python3
"""Shared numerical observation helpers for the fixed GPU replay.

Only writes below refactor_checks/. No production data, preprocessing or main
entrypoint is executed. This is a numerical fixture, not a resume facility.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "refactor_checks/.runtime/matplotlib"))

import numpy as np
import torch
import faiss
from model.stage_model import StageMultiModalModel, should_update_ot
from model import sparse_uot
from model.loss import compute_joint, crossview_contrastive_Loss
from scripts import run_mousebrain as mouse
from scripts import run_crc_stereocite as crc

GIT = "/home/hujinlan/miniconda3/envs/cosie/bin/git"
EXPECTED_HEAD = "922d1738922e8b94890a54f5e42bbf6551f5ccc0"
OUTPUT_KEYS = (
    "latent_dict", "fused_embeddings", "graphsage_embeddings", "ot_embeddings",
    "final_embeddings", "reconstructions", "spatial_graph_dict", "losses",
    "loss_details",
)
TOLERANCES = {
    "atol": 1e-5, "rtol": 1e-5,
    "integers_booleans_shapes_keys_none": "exact",
    "nonfinite": "always fails numerical comparison; known R1 probe separate",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output([GIT, *args], cwd=ROOT, text=True).strip()


def json_write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def snap(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: snap(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snap(v) for v in value)
    return copy.deepcopy(value)


def stable(value):
    # Only wall-clock metadata is excluded, never support/cost/backend values.
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items()
                if not (isinstance(k, str) and
                        (k.endswith("time_sec") or k == "elapsed_time_sec"))}
    if isinstance(value, (list, tuple)):
        return type(value)(stable(v) for v in value)
    return snap(value)


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment():
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("torch", "numpy", "scipy", "scikit-learn", "anndata", "scanpy")},
        "faiss": faiss.__version__, "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "device_index": 0,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_threads": torch.get_num_threads(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "precision": "FP32, no autocast; BF16/CPU numerical coverage not claimed",
    }


def provenance():
    names = git("ls-files").splitlines()
    hashes = {}
    for name in names:
        top = name.split("/")[0]
        if top in ("results", "report_derived_metrics", "data", "UNI",
                   "preprocessed_cache") or top.startswith("result_"):
            continue
        p = ROOT / name
        if p.is_file() and p.stat().st_size < 1000000:
            hashes[name] = sha(p)
    return {
        "head": git("rev-parse", "HEAD"), "status": git("status", "--short"),
        "tracked_diff": git("diff", "--binary"),
        "staged_diff": git("diff", "--cached", "--binary"),
        "tracked_small_file_hashes": hashes,
        "script_sha256": sha(__file__),
    }




def static_refresh_audit():
    result = {}
    for filename, function, forward_name in (
        ("training/fit.py", "iter_fit_model", "run_one_forward"),
    ):
        text = (ROOT / filename).read_text()
        tree = ast.parse(text)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and
                    isinstance(n.target, ast.Name) and n.target.id == "epoch")
        guards = [n for n in ast.walk(loop) if isinstance(n, ast.If) and
                  isinstance(n.test, ast.BoolOp) and isinstance(n.test.op, ast.And)
                  and isinstance(n.test.values[0], ast.Name) and n.test.values[0].id == "refresh_ot"
                  and isinstance(n.test.values[1], ast.Call) and isinstance(n.test.values[1].func, ast.Name)
                  and n.test.values[1].func.id == "should_update_ot"]
        assert len(guards) == 1
        guard = guards[0]
        calls = [n for n in ast.walk(guard) if isinstance(n, ast.Call)]
        forwards = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == forward_name]
        updates = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "update_model_ot_prior"]
        evals = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "eval"]
        nograd = [n for n in ast.walk(guard) if isinstance(n, ast.With) and
                  any("torch.no_grad()" == ast.unparse(x.context_expr) for x in n.items)]
        assert len(forwards) == len(updates) == len(evals) == len(nograd) == 1
        assert evals[0].lineno < nograd[0].lineno < forwards[0].lineno < updates[0].lineno
        assert forwards[0] in list(ast.walk(nograd[0]))
        assert updates[0] in list(ast.walk(nograd[0]))
        steps = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and n.func.attr == "step"]
        assert steps and max(n.lineno for n in steps) < guard.lineno
        train_calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and
                       isinstance(n.func, ast.Attribute) and n.func.attr == "train"]
        assert len(train_calls) == 1 and train_calls[0].lineno < min(n.lineno for n in steps)
        tail = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and
                n.lineno > loop.end_lineno and isinstance(n.func, ast.Attribute) and n.func.attr == "eval"]
        assert tail
        result[filename] = {
            "status": "STATIC_ONLY for epoch100+ whole-loop refresh/mode restoration",
            "function": function, "loop_line": loop.lineno, "refresh_line": guard.lineno,
            "single_forward_line": forwards[0].lineno, "update_line": updates[0].lineno,
            "train_at_loop_start_line": train_calls[0].lineno,
            "final_eval_line": tail[0].lineno,
            "refresh_ast": ast.dump(guard, include_attributes=False),
        }
    # P4b-2 keeps epoch0 in MouseBrain, then delegates the only epoch loop.
    mouse_tree = ast.parse((ROOT / "scripts/run_mousebrain.py").read_text())
    mouse_run = next(n for n in mouse_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_mousebrain")
    assert not any(isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch" for n in ast.walk(mouse_run))
    dry = [n for n in ast.walk(mouse_run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "model"]
    shared = [n for n in ast.walk(mouse_run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "train_small_crc_model"]
    assert len(dry) == len(shared) == 1 and dry[0].lineno < shared[0].lineno
    assert any(k.arg == "epoch" and isinstance(k.value, ast.Constant) and k.value.value == 0 for k in dry[0].keywords)
    assert any(isinstance(n, ast.With) and dry[0] in list(ast.walk(n)) and any(ast.unparse(i.context_expr) == "torch.no_grad()" for i in n.items) for n in ast.walk(mouse_run))
    result["scripts/run_mousebrain.py"] = {"status": "STATIC_ONLY: independent epoch0 then shared fit", "dry_forward_line": dry[0].lineno, "fit_line": shared[0].lineno}
    epochs = [0, 1, 19, 20, 99, 100, 101, 119, 120, 199, 200, 201]
    actual = {str(e): should_update_ot(e, 20) for e in epochs}
    assert [e for e in epochs if actual[str(e)]] == [100, 120, 200]
    result["scheduler_dynamic"] = actual
    return result




def build_model(bundle, inputs, variant):
    config = copy.deepcopy(bundle["config"])
    model = StageMultiModalModel(config=config, feature_dict=inputs["feature_dict"])
    # D2 approved schema retirement. The loaded bundle was projected by exact
    # historical names; require every remaining parameter, dtype and gradient
    # flag to match before strict loading. Never match by positional iteration.
    current = model.state_dict()
    assert list(current) == list(bundle["parameters"])
    for name, value in current.items():
        expected = bundle["parameters"][name]
        assert value.shape == expected.shape and value.dtype == expected.dtype, name
    named = dict(model.named_parameters())
    assert list(named) == list(bundle["parameter_schema"])
    for name, parameter in named.items():
        schema = bundle["parameter_schema"][name]
        assert list(parameter.shape) == schema["shape"], name
        assert str(parameter.dtype) == schema["dtype"], name
        assert parameter.requires_grad is schema["requires_grad"], name
    model.load_state_dict(bundle["parameters"], strict=True)
    if variant == "zero_dropout":
        # Preserve Sequential names and identical effective weights. Changing
        # constructor dropout to zero removes modules and shifts state_dict keys.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    model.ot_prior = copy.deepcopy(inputs["prior"])
    model._spatial_graph_cache = copy.deepcopy(inputs["initial_spatial_cache"])
    return model


def run_step(bundle, inputs, variant="main", chunked=False, checkpoint=False,
             fit_overrides=None, preflight=True, capture_stages=False):
    """Execute the ORIGINAL shared fit, exactly one epoch. Hooks only observe."""
    model = build_model(bundle, inputs, variant)
    features = {s: {m: x.clone().to("cuda").requires_grad_(True) for m, x in mods.items()}
                for s, mods in inputs["feature_dict"].items()}
    order = inputs["section_order"]
    args = SimpleNamespace(**copy.deepcopy(bundle["runner_args"]))
    args.epochs = 1
    args.lr = bundle["config"]["training"]["lr"]
    args.weight_decay = bundle["config"]["training"]["weight_decay"]
    args.amp_dtype = "none"
    args.training_loss_only = variant == "checkpoint_layout"
    args.decoder_chunk_size = 5 if chunked else 0
    args.ot_attention_source_chunk_size = 7 if chunked else 0
    args.cache_spatial_graphs = variant != "main"
    args.log_every = 0
    args.log_cuda_memory_detail = False
    args.output_dir = str(ROOT / "refactor_checks")
    for name in ("checkpoint_ot_attention", "checkpoint_encoder_fusion",
                 "checkpoint_decoder_chunks", "checkpoint_graph_encoder"):
        setattr(args, name, checkpoint)
    for name, value in (fit_overrides or {}).items():
        setattr(args, name, value)
    assert args.epochs == 1 and args.output_dir == str(ROOT / "refactor_checks")
    observation, events, memory = {}, [], []
    memory_cursor = [0]
    phase = ["preflight"]
    collecting, stage_values, section_losses = [False], {}, {}
    def before_forward(module, a, kw):
        collecting[0] = capture_stages
        stage_values.clear()
        section_losses.clear()
        return a, {**kw, "memory_recorder":
                   lambda stage, detail: memory.append((phase[0], stage, snap(detail)))}
    model.register_forward_pre_hook(before_forward, with_kwargs=True)
    def graph_observer(stage_in, stage_out):
        def observe(module, a, output):
            if collecting[0]:
                values = stage_values.setdefault(stage_in, {})
                section = order[len(values)]
                values[section] = snap(a[0])
                stage_values.setdefault(stage_out, {})[section] = snap(output)
        return observe
    if capture_stages:
        model.graphsage.register_forward_hook(graph_observer("fused", "pre_ot"))
        model.post_ot_graphsage.register_forward_hook(graph_observer("ot", "final"))
    stage_module = sys.modules[StageMultiModalModel.__module__]
    original_crossview = stage_module.compute_pairwise_cosie_crossview_loss
    original_decode = model._decode_and_reconstruct_section
    def observe_crossview(*a, **kw):
        result = original_crossview(*a, **kw)
        if collecting[0]:
            values = section_losses.setdefault("crossview", {})
            values[order[len(values)]] = snap(result)
        return result
    def observe_decode(*a, **kw):
        result = original_decode(*a, **kw)
        if collecting[0]:
            section_losses.setdefault("reconstruction", {})[kw["section"]] = snap(result[:2])
        return result
    def capture(module, a, kw, output):
        collecting[0] = False  # Exclude checkpoint backward recomputation.
        label = "train" if module.training and torch.is_grad_enabled() else (
            "preflight" if module.training else "final_eval")
        observation[label] = snap({k: output[k] for k in OUTPUT_KEYS if k in output})
        if capture_stages:
            assert all(list(stage_values[k]) == order for k in ("fused", "pre_ot", "ot", "final"))
            observation[label]["observed_stages"] = snap(stage_values)
            observation[label]["observed_loss_details"] = snap(section_losses)
            observation[label]["autocast_enabled"] = torch.is_autocast_enabled()
        stages = memory[memory_cursor[0]:]
        memory_cursor[0] = len(memory)
        updates = [i for i, (_, stage, _) in enumerate(stages)
                   if stage.startswith("ot_attention_update_")]
        applies = [(i, detail) for i, (_, stage, detail) in enumerate(stages)
                   if stage.startswith("ot_attention_apply_")]
        assert len(updates) == 2 * (len(order)-1) and len(applies) == len(order)
        assert max(updates) < min(i for i, _ in applies)
        counts = {d["section"]: d["update_count"] for _, d in applies}
        assert counts == {s: 1 if i in (0, len(order)-1) else 2 for i, s in enumerate(order)}
        events.append({"event": "forward", "phase": label, "epoch": kw.get("epoch"),
                       "train": module.training, "grad_enabled": torch.is_grad_enabled(),
                       "directional_update_counts": counts, "all_updates_before_apply": True})
    model.register_forward_hook(capture, with_kwargs=True)
    real_adam = torch.optim.Adam
    def make_adam(*a, **kw):
        opt = real_adam(*a, **kw)
        observation["optimizer_defaults"] = snap(opt.defaults)
        def pre_step(optimizer, a, kw):
            observation["gradients"] = {name: snap(p.grad) for name, p in model.named_parameters()}
            observation["input_gradients"] = {s: {m: snap(x.grad) for m, x in mods.items()}
                                               for s, mods in features.items()}
            events.append({"event": "optimizer_step_pre", "train": model.training,
                           "grad_enabled": torch.is_grad_enabled()})
        def post_step(optimizer, a, kw):
            active = {name for name, grad in observation["gradients"].items() if grad is not None}
            observation["parameters_after"] = {name: snap(p) for name, p in model.named_parameters()
                                                if name in active}
            for name, p in model.named_parameters():
                if name not in active:
                    assert torch.equal(p.detach().cpu(), bundle["parameters"][name])
            observation["rng_after_step"] = snap(rng_state())
            events.append({"event": "optimizer_step_post"})
        opt.register_step_pre_hook(pre_step)
        opt.register_step_post_hook(post_step)
        return opt
    restore_rng(inputs["rng_before_preflight"])
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as observers:
        if capture_stages:
            observers.enter_context(patch.object(stage_module, "compute_pairwise_cosie_crossview_loss", observe_crossview))
            observers.enter_context(patch.object(model, "_decode_and_reconstruct_section", observe_decode))
        if preflight:
            model.train()
            with torch.no_grad():
                model(feature_dict=features, spatial_loc_dict=inputs["spatial_loc_dict"],
                      section_order=order, epoch=0)
        phase[0] = "fit"
        with patch.object(torch.optim, "Adam", make_adam):
            history, final, updated_epochs = crc.train_small_crc_model(
                model, features, inputs["spatial_loc_dict"], None, order, args)
    assert updated_epochs == []
    assert len(history) == 1
    observation["history"] = stable(history)
    observation["prior_after_step"] = stable(model.ot_prior)
    observation["spatial_cache_after"] = snap(model._spatial_graph_cache)
    observation["rng_after_fit"] = snap(rng_state())
    active_flags = {
        name: any(detail.get(name, False) for _, _, detail in memory)
        for name in ("checkpoint_encoder_fusion", "graph_encoder_checkpoint_active",
                     "checkpoint_ot_attention", "decoder_chunk_checkpoint_active")
    }
    if checkpoint:
        assert all(active_flags.values()), active_flags
    meta = {
        "variant": variant,
        "dropout_override": 0.0 if variant == "zero_dropout" else None,
        "mode_events": events, "checkpoint_branch_flags": active_flags,
        "active_parameter_count": sum(g is not None for g in observation["gradients"].values()),
        "grad_none_names": [n for n, g in observation["gradients"].items() if g is None],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "fit_runtime_args": vars(args), "memory_branch_events": memory,
    }
    # Return actual outputs/model for the independent real refresh-helper probe.
    return observation, meta, model, final


def sparse_probe(model, inputs, args_dict, final=None, helper_module=mouse):
    captures, extractions = [], []
    real_solver = sparse_uot.sparse_unbalanced_sinkhorn_bidirectional_topk
    real_extract = sparse_uot._edges_to_sparse_topk
    pointers = []
    def extract(**kw):
        extractions.append(snap({k: kw[k] for k in
                                ("source_ids", "target_ids", "mass", "n_source", "n_target")}))
        pointers.append(kw["mass"].data_ptr())
        return real_extract(**kw)
    def solver(**kw):
        captures.append(snap(kw))
        return real_solver(**kw)
    args = SimpleNamespace(**args_dict)
    preparation = {}
    original_prepare = model.prepare_ot_prior_refresh
    def prepare(*a, **kw):
        embeddings, contexts, diagnostics = original_prepare(*a, **kw)
        assert diagnostics["refresh_source"] == "ot"
        assert diagnostics["topology_context_weight"] == 0.2
        for s in inputs["section_order"]:
            assert torch.equal(embeddings[s], final["ot_embeddings"][s])
            assert not embeddings[s].requires_grad and not contexts[s].requires_grad
        preparation.update(snap({"embeddings": embeddings, "contexts": contexts,
                                 "diagnostics": diagnostics,
                                 "grad_enabled": torch.is_grad_enabled(),
                                 "model_training": model.training}))
        return embeddings, contexts, diagnostics
    with patch.object(sparse_uot, "sparse_unbalanced_sinkhorn_bidirectional_topk", solver), \
         patch.object(sparse_uot, "_edges_to_sparse_topk", extract):
        if final is None:
            prior = helper_module.initialize_model_ot_prior(
                model, inputs["feature_dict"], inputs["section_order"], args)
        else:
            old = model.ot_prior
            assert not model.training
            with patch.object(model, "prepare_ot_prior_refresh", prepare), torch.no_grad():
                prior = helper_module.update_model_ot_prior(model, final, inputs["section_order"], args)
            assert model.ot_prior is prior and prior is not old and not model.training
    assert len(captures) == len(inputs["section_order"]) - 1
    assert len(extractions) == 2 * len(captures)
    for call in captures:
        assert bool(((call["edge_src"] >= 0) & (call["edge_src"] < call["n_source"])).all())
        assert bool(((call["edge_tgt"] >= 0) & (call["edge_tgt"] < call["n_target"])).all())
    for i in range(0, len(extractions), 2):
        a, b = extractions[i:i+2]
        assert pointers[i] == pointers[i+1]  # The original solver's identical p_edge.
        assert torch.equal(a["mass"], b["mass"])
        assert torch.equal(a["source_ids"], b["target_ids"])
        assert torch.equal(a["target_ids"], b["source_ids"])
    expected_keys = {(a,b) for a,b in zip(inputs["section_order"][:-1], inputs["section_order"][1:])}
    expected_keys |= {(b,a) for a,b in list(expected_keys)}
    assert set(prior) == expected_keys
    for (source, target), p in prior.items():
        n = len(inputs["spot_order"][source]); nt = len(inputs["spot_order"][target])
        assert p["topk_idx"].shape == (n, min(10, nt))
        assert bool(((p["topk_idx"] >= 0) & (p["topk_idx"] < nt)).all())
    return stable({"prior": prior, "solver_inputs": captures,
                   "coupling_extractions": extractions, "preparation": preparation})


def fixed_support_probe():
    kwargs = {
        "edge_src": torch.tensor([0, 0, 1, 1, 2, 4, 6]),
        "edge_tgt": torch.tensor([0, 3, 1, 7, 4, 10, 5]),
        "edge_cost": torch.tensor([0.1, 1.2, 0.4, 0.3, 1.0, 0.2, 0.8]),
        "n_source": 7, "n_target": 11, "epsilon": .05, "tau_a": 1.,
        "tau_b": 1., "max_iter": 100, "stabilizer": 1e-8,
        "attention_topk": 10, "device": "cuda",
    }
    result = sparse_uot.sparse_unbalanced_sinkhorn_bidirectional_topk(**kwargs)
    assert torch.count_nonzero(result["left_to_right"]["topk_weight"][3]) == 0
    empty = {**kwargs, "edge_src": torch.empty(0, dtype=torch.long),
             "edge_tgt": torch.empty(0, dtype=torch.long), "edge_cost": torch.empty(0)}
    try:
        sparse_uot.sparse_unbalanced_sinkhorn_bidirectional_topk(**empty)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("Empty support no longer raises ValueError")
    return snap({"input": kwargs, "output": result, "empty_support_error": error})


def known_r1_probe():
    z = torch.tensor([[1., -1.], [1., -1.]], device="cuda") / math.sqrt(2)
    joint = compute_joint(z, z)
    loss = crossview_contrastive_Loss(z, z, gamma=5.)
    return snap({"input": z, "joint": joint, "loss": loss,
                 "loss_is_finite": bool(torch.isfinite(loss)),
                 "status": "KNOWN_FAILURE" if not torch.isfinite(loss) else "CHANGED"})


def compare(actual, expected, stats, errors, path, tolerance):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            errors.append(f"{path}: keys differ"); return
        for key in expected:
            compare(actual[key], expected[key], stats, errors, f"{path}/{key}", tolerance)
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            errors.append(f"{path}: sequence schema differs"); return
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, stats, errors, f"{path}/{i}", tolerance)
    elif isinstance(expected, (torch.Tensor, np.ndarray)):
        a = torch.as_tensor(actual); e = torch.as_tensor(expected)
        if a.shape != e.shape or a.dtype != e.dtype:
            errors.append(f"{path}: tensor shape/dtype differs"); return
        group = next((k for k in ("gradients", "parameters_after", "losses", "loss_details",
                     "solver_inputs", "coupling_extractions", "prior", "spatial_graph_dict")
                      if k in path), "embeddings_and_state")
        entry = stats.setdefault(group, {"tensor_count": 0, "max_abs": 0., "max_rel": 0.})
        entry["tensor_count"] += 1
        if a.is_floating_point():
            if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):
                errors.append(f"{path}: nonfinite tensor"); return
            delta = (a.double()-e.double()).abs()
            entry["max_abs"] = max(entry["max_abs"], float(delta.max()) if delta.numel() else 0.)
            entry["max_rel"] = max(entry["max_rel"],
                float((delta/e.double().abs().clamp_min(1e-12)).max()) if delta.numel() else 0.)
            if not torch.allclose(a, e, atol=tolerance["atol"], rtol=tolerance["rtol"]):
                errors.append(f"{path}: numerical mismatch max_abs={float(delta.max()):.8g}")
        elif not torch.equal(a, e):
            errors.append(f"{path}: integer/RNG value differs")
    elif isinstance(expected, float):
        if not (math.isfinite(actual) and math.isfinite(expected)):
            errors.append(f"{path}: nonfinite scalar")
        elif not math.isclose(actual, expected, abs_tol=tolerance["atol"], rel_tol=tolerance["rtol"]):
            errors.append(f"{path}: scalar mismatch")
    elif actual != expected:
        errors.append(f"{path}: value differs {actual!r} != {expected!r}")









def equivalent_numeric(obs):
    # Cache population differs by execution option; graph CONTENT is compared.
    return {k: v for k, v in obs.items() if k not in
            ("spatial_cache_after", "rng_after_step", "rng_after_fit")}
