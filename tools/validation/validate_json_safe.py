"""Regression for SPATCH's completed-run summary serialization."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from training.fit import json_safe


class JsonSafeTests(unittest.TestCase):
    def test_nested_runner_paths(self):
        summary = {"resolved_config": {"runner": {
            "output_dir": Path("/tmp/run"),
            "preprocessed_cache_dir": Path("/tmp/cache"),
            "optional": None,
        }}, "paths": (Path("a"), {"b": Path("b")})}
        saved = json.loads(json.dumps(json_safe(summary)))
        self.assertEqual(saved["resolved_config"]["runner"]["output_dir"], "/tmp/run")
        self.assertEqual(saved["paths"], ["a", {"b": "b"}])
        self.assertIsInstance(summary["resolved_config"]["runner"]["output_dir"], Path)

    def test_existing_numeric_contract(self):
        data = {"scalar": torch.tensor(2.), "tensor": torch.ones(2, 3),
                "array": np.array([1, 2]), "numpy": np.int64(3)}
        self.assertEqual(json.loads(json.dumps(json_safe(data))),
                         {"scalar": 2., "tensor": [2, 3], "array": [1, 2], "numpy": 3})

    def test_real_spatch_config_serializes_without_loading_data(self):
        from experiments.v15c2.plan import load_suite, render_job
        from scripts.run_spatch import parse_args, resolve_run_config
        suite = load_suite()
        ds = next(d for d in suite["datasets"] if d["id"] == "spatch")
        args = parse_args(render_job(suite, ds, 42, "train")["command"][2:])
        resolved = resolve_run_config(args)
        saved = json.loads(json.dumps(json_safe({"resolved_config": resolved})))
        self.assertEqual(saved["resolved_config"]["model_config"]["training"]["device"], "cuda")
        self.assertEqual(saved["resolved_config"]["runner"]["seed"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
