"""CPU unit checks only: no dataset loading, training loop or analysis runs."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from model.configure import get_default_model_config
from model.feature_graph import FeatureGraphCache, compute_feature_knn_graph
from model.stage_model import StageMultiModalModel, should_update_ot
from training.graph_refresh import prepare_refresh_outputs

ROOT = Path(__file__).resolve().parents[2]


class FeatureGraphTests(unittest.TestCase):
    def test_exact_cosine_neighbours_and_detachment(self):
        torch.manual_seed(8)
        z = torch.randn(17, 9, requires_grad=True)
        edge = compute_feature_knn_graph(z, 3)
        norm = torch.nn.functional.normalize(z.detach(), dim=1)
        similarities = norm @ norm.T
        similarities.fill_diagonal_(-float("inf"))
        expected = similarities.topk(3, dim=1).indices
        self.assertTrue(torch.equal(edge[1].view(17, 3), expected))
        self.assertFalse(edge.requires_grad)
        self.assertFalse(torch.any(edge[0] == edge[1]))

    def test_combined_weights_preserve_duplicate_edges(self):
        cache = FeatureGraphCache(enabled=True, k_spatial=5)
        self.assertEqual(cache.k_feature, 3)
        cache.edges = {"A": torch.tensor([[0, 1], [1, 0]])}
        cache.n_spots = {"A": 2}
        spatial = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        weights = torch.tensor([.25, .75, .4, .6])
        edges, actual = cache.combine("A", torch.zeros(2, 4), spatial, weights)
        self.assertEqual(edges.shape[1], 6)
        torch.testing.assert_close(actual, torch.tensor([.25, .75, .4, .6, .2, .2]) / 1.2)
        # The spatial tensors are preserved for post-OT and topology context.
        torch.testing.assert_close(weights, torch.tensor([.25, .75, .4, .6]), rtol=0, atol=0)

    def test_atomic_cache_refresh(self):
        cache = FeatureGraphCache(enabled=True, k_spatial=5)
        cache.refresh({"A": torch.randn(8, 4)})
        old = cache.edges
        with self.assertRaises(ValueError):
            cache.refresh({"A": torch.randn(8, 4), "B": torch.randn(2, 4)})
        self.assertIs(cache.edges, old)
        self.assertEqual(cache.generation, 1)

    def test_schedule(self):
        self.assertEqual([e for e in range(1, 201) if should_update_ot(e, 20)],
                         [100, 120, 140, 160, 180, 200])

    def test_disabled_has_no_build_or_extra_pass(self):
        from types import SimpleNamespace
        model = SimpleNamespace(training=False, feature_graph=FeatureGraphCache(enabled=False, k_spatial=5))
        forward = unittest.mock.Mock(return_value={"sentinel": 1})
        with torch.no_grad(), patch("model.feature_graph.compute_feature_knn_graph", side_effect=AssertionError):
            output, event = prepare_refresh_outputs(model, forward, epoch=100)
        self.assertEqual(output, {"sentinel": 1})
        self.assertIsNone(event)
        self.assertEqual(forward.call_count, 1)

    def test_refresh_pass_order(self):
        from types import SimpleNamespace
        graph = FeatureGraphCache(enabled=True, k_spatial=3)
        model = SimpleNamespace(training=False, feature_graph=graph)
        passes = []

        def forward():
            passes.append(graph.generation)
            return {"fused_embeddings": {"A": torch.randn(8, 4)},
                    "ot_embeddings": {"A": graph.generation}}

        with torch.no_grad():
            outputs, event = prepare_refresh_outputs(model, forward, epoch=100)
            outputs2, event2 = prepare_refresh_outputs(model, forward, epoch=120)
        self.assertEqual(passes, [0, 1, 1, 2])
        self.assertEqual(outputs["ot_embeddings"]["A"], 1)
        self.assertEqual(outputs2["ot_embeddings"]["A"], 2)
        self.assertTrue(event["cache_empty_before_refresh"])
        self.assertFalse(event2["cache_empty_before_refresh"])
        with self.assertRaises(RuntimeError):
            prepare_refresh_outputs(model, forward, epoch=140)

    def test_stage_baseline_and_pre_post_separation(self):
        path = ROOT / "refactor_checks/v15c2_implementation_20260919/before_sources/model/stage_model.py"
        spec = importlib.util.spec_from_file_location("model._feature_graph_baseline", path)
        baseline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_module)
        torch.manual_seed(99)
        features = {s: {"RNA": torch.randn(n, 6), "Protein": torch.randn(n, 4)}
                    for s, n in (("A", 9), ("B", 11))}
        spatial = {s: torch.rand(len(mods["RNA"]), 2) for s, mods in features.items()}
        config = get_default_model_config()
        config["training"]["device"] = "cpu"
        config["graph"]["knn_neighbors_spatial"] = 3
        torch.manual_seed(42)
        baseline = baseline_module.StageMultiModalModel(config, features).eval()
        old_rng = torch.get_rng_state()
        torch.manual_seed(42)
        current = StageMultiModalModel(config, features).eval()
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(list(baseline.state_dict()), list(current.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[key], atol=0, rtol=0)
        baseline.initialize_candidate_sparse_ot_prior(
            features, section_order=["A", "B"], initial_modality_candidate_k=4,
            candidate_k=5, attention_topk=3, candidate_backend="blockwise",
            faiss_device="cpu", max_iter=4,
        )
        current.ot_prior = copy.deepcopy(baseline.ot_prior)
        kw = dict(feature_dict=features, spatial_loc_dict=spatial, section_order=["A", "B"], epoch=100)
        with torch.no_grad():
            old = baseline(**kw)
            disabled = current(**kw)
            for s in features:
                torch.testing.assert_close(old["final_embeddings"][s], disabled["final_embeddings"][s], atol=0, rtol=0)
            current.feature_graph = FeatureGraphCache(enabled=True, k_spatial=3)
            before = current(**kw)
            self.assertEqual(current.feature_graph.generation, 0)
            for s in features:
                torch.testing.assert_close(old["final_embeddings"][s], before["final_embeddings"][s], atol=0, rtol=0)
            captured = []
            hook = current.post_ot_graphsage.register_forward_pre_hook(lambda module, args: captured.append(args[1].clone()))
            refreshed, event = prepare_refresh_outputs(current, lambda: current(**kw), epoch=100)
            hook.remove()
        self.assertEqual(len(captured), 4)
        for i, s in enumerate(features):
            self.assertTrue(torch.equal(captured[i + 2], before["spatial_graph_dict"][s]["edge_index"]))
        self.assertGreater(float((refreshed["graphsage_embeddings"]["A"] - before["graphsage_embeddings"]["A"]).abs().max()), 0)

    def test_new_analysis_root_is_explicit_and_scoped(self):
        from analysis.cache import CacheMismatch, check_output_path, writable_result_root
        root = ROOT / "result_v15C-2"
        target = root / "mousebrain/seed_42/analysis_attempt_01"
        with self.assertRaises(CacheMismatch):
            check_output_path(target)
        with writable_result_root(root):
            self.assertEqual(check_output_path(target), target)
            with self.assertRaises(CacheMismatch):
                check_output_path(ROOT / "result_v15C/analysis")
            with self.assertRaises(CacheMismatch):
                check_output_path(target, [target])
            with self.assertRaises(CacheMismatch):
                check_output_path(root / "preprocessed_cache/analysis")
        with self.assertRaises(CacheMismatch):
            check_output_path(target)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
