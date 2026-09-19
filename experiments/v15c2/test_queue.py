"""Planning/queue unit tests. All child processes are mocked; no training."""
from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import plan, supervisor
from .run import main


class QueueTests(unittest.TestCase):
    def test_gpu_retries_until_ready_and_never_falls_back(self):
        from types import SimpleNamespace
        from .gpu import probe_gpu
        fail = SimpleNamespace(returncode=1, stdout="", stderr="CUDA invisible")
        ready = SimpleNamespace(returncode=0, stdout='{"status":"ready","gpu":"mock"}', stderr="")
        with patch("experiments.v15c2.gpu.subprocess.run", side_effect=[fail, fail, ready]) as run, patch("experiments.v15c2.gpu.time.sleep") as sleep, contextlib.redirect_stderr(io.StringIO()):
            report = probe_gpu("python")
            self.assertEqual(report["attempt"], 3)
            self.assertEqual(len(report["previous_failures"]), 2)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
        with patch("experiments.v15c2.gpu.subprocess.run", return_value=fail) as run, patch("experiments.v15c2.gpu.time.sleep"), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback, no training started"):
                probe_gpu("python")
            self.assertEqual(run.call_count, 3)

    def test_required_gpu_and_spatch_cache_contract(self):
        suite = plan.load_suite()
        report = plan.check_inputs(suite)
        self.assertEqual(report["training_jobs"], 18)
        self.assertEqual(report["analysis_jobs"], 18)
        cpu_suite = copy.deepcopy(suite)
        args = cpu_suite["datasets"][0]["train_argv"]
        args[args.index("--device") + 1] = "cpu"
        with self.assertRaisesRegex(ValueError, "requires GPU"):
            plan.check_inputs(cpu_suite)
        bad_cache = copy.deepcopy(suite)
        spatch = next(d for d in bad_cache["datasets"] if d["id"] == "spatch")
        spatch["train_argv"].append("--build_preprocessed_cache")
        with self.assertRaisesRegex(ValueError, "must not be rebuilt"):
            plan.check_inputs(bad_cache)

    def test_lock_rejects_duplicate_supervisor(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "_control"
            control.mkdir()
            with (control / "suite.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(supervisor, "check_inputs"), patch.object(supervisor, "output_root", return_value=root), patch.object(supervisor, "probe_gpu", return_value={"status": "ready", "gpu": "mock"}), patch.object(supervisor, "_run_locked", side_effect=AssertionError("must not execute")):
                    with self.assertRaisesRegex(RuntimeError, "owns the queue"):
                        supervisor.run(plan.load_suite())

    def test_plan_is_read_only_and_has_18_independent_pairs(self):
        with patch("subprocess.Popen", side_effect=AssertionError("No launch in plan mode")), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main([]), 0)
        jobs = json.loads(out.getvalue())["jobs"]
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len({j["output_dir"] for j in jobs}), 36)
        for train, analyze in zip(jobs[::2], jobs[1::2]):
            self.assertEqual(analyze["depends_on"], train["id"])
            self.assertEqual(analyze["run_dir"], train["output_dir"])
        self.assertEqual({j["seed"] for j in jobs}, {42, 43, 44})

    def test_all_18_runner_and_analysis_configs_match_history(self):
        from scripts.evaluate import parse_args as parse_analysis
        from analysis import protocols
        suite = plan.load_suite()
        compared = 0
        for ds in suite["datasets"]:
            module = importlib.import_module("scripts.run_" + ds["id"])
            baseline = json.loads((plan.ROOT / ds["baseline_summary"]).read_text())
            old_analysis = json.loads((plan.ROOT / ds["baseline_analysis_config"]).read_text())
            old_k = old_analysis.get("kmeans", old_analysis.get("clustering"))
            for seed in suite["seeds"]:
                args = module.parse_args(plan.render_job(suite, ds, seed, "train")["command"][2:])
                # Only bypass the resolver's device availability guard. No model,
                # tensor, data loader, main(), fit() or evaluate() is called.
                with patch("torch.cuda.is_available", return_value=True):
                    if ds["id"] == "mousebrain":
                        config, args = module.resolve_run_config(json.loads(Path(args.config).read_text()), args)
                    elif ds["id"] == "mouse_thymus":
                        config = module.misar.make_model_config(args, "Protein")
                    elif ds["id"] in ("mouse_spleen", "human_lymph_node"):
                        config = module.crc.build_model_config(args)
                    elif ds["id"] == "misar_seq":
                        config = module.make_model_config(args)
                    else:
                        config = module.resolve_run_config(args)["model_config"]
                self.assertTrue(config["feature_graph"]["enabled"])
                self.assertEqual(args.device, "cuda")
                self.assertEqual(config["training"]["device"], "cuda")
                self.assertEqual(args.seed, seed)
                self.assertEqual(config["graphsage"]["post_ot_graphsage_scale"], .5)
                for key, value in vars(args).items():
                    if key in baseline and isinstance(baseline[key], (int, float, str, bool, type(None))) and key not in ("seed", "output_dir", "input_data_path"):
                        self.assertEqual(value, baseline[key], (ds["id"], seed, key))
                analysis_args = parse_analysis(plan.render_job(suite, ds, seed, "analysis")["command"][2:])
                self.assertEqual(analysis_args.k, old_k.get("joint_k_values", old_k.get("k_values")))
                self.assertEqual(analysis_args.protocol, "requested")
                if ds["id"] != "spatch":
                    spec = protocols.SPECS[ds["id"]]
                    self.assertEqual((spec.seed, spec.n_init, spec.max_iter), (old_k["seed"], old_k["n_init"], old_k["max_iter"]))
                    self.assertEqual(spec.batch_max_samples, old_analysis["batch_metric_parameters"]["max_samples"])
                compared += 1
        self.assertEqual(compared, 18)

    def test_continuation_resume_and_fresh_retry_directories(self):
        suite = plan.load_suite()
        launches = []
        failed_once = False
        original_render = plan.render_job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "_control"
            control.mkdir()

            def render(suite, ds, seed, stage, attempt=1, trained_dir=None):
                job = original_render(suite, ds, seed, stage, attempt, trained_dir)
                for field in ("output_dir", "run_dir"):
                    job[field] = job[field].replace(str(plan.output_root(suite)), str(root))
                job["command"] = [job["id"], job["output_dir"]]
                return job

            class FakeProcess:
                pid = 99999999

                def __init__(self, command, **kwargs):
                    nonlocal failed_once
                    launches.append(command)
                    Path(command[1]).mkdir(parents=True)
                    self.code = 0
                    if command[0] == "misar_seq/seed_42/train" and not failed_once:
                        self.code = 7
                        failed_once = True

                def wait(self, timeout=None):
                    return self.code

                def poll(self):
                    return self.code

            with patch.object(supervisor, "source_identity", return_value={"sha256": "unit-test", "files": {}}), patch.object(supervisor, "render_job", side_effect=render), patch.object(supervisor.subprocess, "Popen", FakeProcess), patch.object(supervisor, "artifacts", return_value={"unit_fixture": True}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(supervisor._run_locked(suite, control, False, False, "mocked"), 1)
                self.assertEqual(len(launches), 35)
                self.assertNotIn("misar_seq/seed_42/analysis", [x[0] for x in launches])
                state = json.loads((control / "status.json").read_text())
                self.assertEqual(state["status"], "completed_with_failures")
                self.assertEqual(supervisor._run_locked(suite, control, True, False, "mocked"), 0)
                self.assertEqual(len(launches), 37)
                self.assertIn("train_attempt_02", launches[-2][1])
                self.assertEqual(launches[-1][0], "misar_seq/seed_42/analysis")
                self.assertEqual(supervisor._run_locked(suite, control, True, False, "mocked"), 0)
                self.assertEqual(len(launches), 37)
                with self.assertRaises(RuntimeError):
                    supervisor._run_locked(suite, control, False, False, "mocked")

    def test_output_validation_covers_dataset_schemas(self):
        suite = plan.load_suite()
        with tempfile.TemporaryDirectory() as directory:
            for ds in suite["datasets"]:
                out = Path(directory) / ds["id"]
                out.mkdir()
                job = plan.render_job(suite, ds, 42, "train")
                job["output_dir"] = str(out)
                summary = {"epochs": 200, "seed": 42, "ot_updates": [100, 120, 140, 160, 180, 200],
                           "resolved_config": {"model_config": {"feature_graph": {"enabled": True}, "training": {"device": "cuda"}}},
                           "total_loss_finite": True, "training_history": [{"total_loss": 1.0}] * 200}
                (out / "run_summary.json").write_text(json.dumps(summary))
                (out / "feature_graph_refresh.json").write_text(json.dumps([
                    {"epoch": e, "refresh_passes_before_uot": 2,
                     "uot_refresh_embeddings_from_updated_feature_graph": True}
                    for e in summary["ot_updates"]]))
                count = 3 if ds["id"] == "mousebrain" else (4 if ds["id"] in ("misar_seq", "mouse_thymus") else 2)
                for i in range(count):
                    path = out / f"final_embeddings_{i}.npy"
                    if ds["id"] == "mousebrain":
                        path = out / "final_embeddings" / f"s{i}_final_embedding.npy"
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"unit-test placeholder; no arrays loaded")
                self.assertEqual(len(supervisor.artifacts(job)), count + 2)
                path.unlink()
                with self.assertRaises(ValueError):
                    supervisor.artifacts(job)


if __name__ == "__main__":
    unittest.main(verbosity=2)
