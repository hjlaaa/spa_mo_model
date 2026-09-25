"""Run: python -m unittest tools.validation.test_interaction_neighbor -v."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import torch

from model.model_component import OTGuidedAttention, spatial_pool_self_excluded
from model.stage_model import StageMultiModalModel
from training.config import resolve_model_config, serialize_config
from tools.validation.validate_microenvironment_context import make_inputs, make_config, initialize_prior


def pool(h, edges, weights):
    return spatial_pool_self_excluded(h, edges, weights, edge_batch_size=2,
                                      l2_normalize=False, detach=False)


class LegacyAttention(OTGuidedAttention):
    """Frozen pre-change chunk formula, independent of the optional value input."""
    def _compute_update_chunk(self, source_h, target_h, topk_idx, topk_weight,
                              confidence, epoch=None, target_value=None):
        q = self.W_Q(source_h)
        candidate_h = target_h[topk_idx]
        k = self.W_K(candidate_h)
        v = self.W_V(candidate_h)
        scores = (q.unsqueeze(1) * k).sum(dim=-1) / math.sqrt(self.d_attn)
        log_prior = torch.log(topk_weight.float().clamp_min(self.delta))
        scores = scores.float() + self._current_beta(epoch) * log_prior
        alpha = torch.softmax(scores, dim=1).to(v.dtype)
        message = (alpha.unsqueeze(-1) * v).sum(dim=1)
        message_bar = self.W_O(message)
        gate = self.gate_mlp(torch.cat([source_h, message_bar,
                                      source_h - message_bar, source_h * message_bar], dim=-1))
        scale = confidence.unsqueeze(-1) * gate if self.use_confidence else gate
        return scale * message_bar


class NeighborTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(712)
        torch.set_num_threads(1)
        self.edges = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3],
                                   [0, 1, 2, 0, 1, 0, 2, 3]])
        self.weights = torch.tensor([.2, .3, .5, .7, .3, .6, .4, 1.])

    def test_pool_mean_self_exclusion_gradient_and_empty(self):
        h = torch.tensor([[1., 0.], [0., 2.], [3., 0.], [7., 8.]], requires_grad=True)
        n = pool(h, self.edges, self.weights)
        torch.testing.assert_close(n, torch.tensor([[1.875, .75], [1., 0.], [1., 0.], [0., 0.]]))
        n[0].sum().backward()
        torch.testing.assert_close(h.grad, torch.tensor([[0., 0.], [.375, .375], [.625, .625], [0., 0.]]))
        for edges, weights in [(self.edges[:, -1:], self.weights[-1:]),
                                (self.edges[:, :0], self.weights[:0]),
                                (self.edges, torch.zeros_like(self.weights))]:
            empty = pool(h, edges, weights)
            self.assertTrue(torch.equal(empty, torch.zeros_like(h)))
            self.assertTrue(torch.isfinite(empty).all())
        self.assertFalse(spatial_pool_self_excluded(h, self.edges, self.weights).requires_grad)

    def test_zero_matches_legacy_output_gradients_rng(self):
        current = OTGuidedAttention(dim=8, d_attn=8, dropout=.2)
        legacy = LegacyAttention(dim=8, d_attn=8, dropout=.2)
        legacy.load_state_dict(current.state_dict())
        a, b = torch.randn(7, 8), torch.randn(11, 8)
        idx = torch.randint(11, (7, 3)); weight = torch.rand(7, 3); conf = torch.rand(7)
        for checkpoint in [False, True]:
            results = []
            for module in [legacy, current]:
                module.zero_grad(set_to_none=True)
                x, y = a.clone().requires_grad_(), b.clone().requires_grad_()
                torch.manual_seed(22)
                out = module(x, y, idx, weight, conf, source_chunk_size=2,
                             checkpoint_attention=checkpoint)
                out.square().sum().backward()
                results.append((out, x.grad, y.grad, torch.get_rng_state(),
                                *[p.grad for p in module.parameters()]))
            for old, new in zip(*results):
                self.assertTrue(torch.equal(old, new))

    def test_neighbor_only_value_gradient_and_qk_unchanged(self):
        module = OTGuidedAttention(dim=8, d_attn=8, dropout=0., norm=None)
        source = torch.randn(1, 8)
        h = torch.randn(4, 8, requires_grad=True)
        # Attention selects only target 0; its spatial neighbors 1 and 2 must get gradients.
        idx = torch.zeros(1, 1, dtype=torch.long)
        records = {}
        hooks = [getattr(module, name).register_forward_pre_hook(
            lambda m, args, name=name: records.setdefault(name, []).append(args[0].detach().clone()))
            for name in ['W_Q', 'W_K', 'W_V']]
        module(source, h, idx, torch.ones(1, 1), torch.ones(1))
        n = pool(h, self.edges, self.weights)
        value = .75 * h + .25 * n
        out = module(source, h, idx, torch.ones(1, 1), torch.ones(1), target_value=value)
        out.square().sum().backward()
        for hook in hooks: hook.remove()
        for name in ['W_Q', 'W_K']:
            self.assertTrue(torch.equal(*records[name]))
        torch.testing.assert_close(records['W_V'][1], value[idx])
        self.assertGreater(h.grad[1:3].abs().sum().item(), 0.)
        self.assertEqual(h.grad[3].abs().sum().item(), 0.)

    def test_dtype_device_chunk_checkpoint_sparse(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in [torch.float32, torch.bfloat16]:
                with self.subTest(device=device, dtype=dtype):
                    a = torch.randn(7, 8, device=device, dtype=dtype, requires_grad=True)
                    b = torch.randn(4, 8, device=device, dtype=dtype, requires_grad=True)
                    n = pool(b, self.edges.to(device), self.weights.to(device))
                    self.assertEqual((n.shape, n.dtype, n.device), (b.shape, b.dtype, b.device))
                    value = .75 * b + .25 * n
                    attn = OTGuidedAttention(dim=8, d_attn=8, dropout=0.).to(device=device, dtype=dtype)
                    idx = torch.randint(4, (7, 2), device=device)
                    weight = torch.rand(7, 2, device=device); conf = torch.ones(7, device=device)
                    full = attn(a, b, idx, weight, conf, target_value=value)
                    shapes = []
                    hooks = [layer.register_forward_pre_hook(lambda m, args: shapes.append(tuple(args[0].shape)))
                             for layer in [attn.W_K, attn.W_V]]
                    chunk = attn(a, b, idx, weight, conf, source_chunk_size=2,
                                 checkpoint_attention=True, target_value=value)
                    grad_full = torch.autograd.grad(full.float().square().sum(), (a, b), retain_graph=True)
                    grad_chunk = torch.autograd.grad(chunk.float().square().sum(), (a, b))
                    for hook in hooks: hook.remove()
                    tol = .04 if dtype == torch.bfloat16 else 1e-5
                    torch.testing.assert_close(full, chunk, atol=tol, rtol=tol)
                    for x, y in zip(grad_full, grad_chunk):
                        torch.testing.assert_close(x, y, atol=tol, rtol=tol)
                    self.assertTrue(all(s[0] <= 2 and s[1:] == (2, 8) for s in shapes))
                    self.assertEqual((chunk.dtype, chunk.device), (a.dtype, a.device))
                    self.assertTrue(torch.isfinite(grad_chunk[1]).all())

    def test_sparse_allocations_and_amp(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten

        class NoDensePairs(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                for tensor in tree_flatten(result)[0]:
                    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                        if tuple(tensor.shape[:2]) in {(7, 4), (4, 7), (7, 7), (4, 4)}:
                            raise AssertionError(f"Dense pair tensor: {func}, {tensor.shape}")
                return result

        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            h = torch.randn(4, 8, device=device, requires_grad=True)
            a = torch.randn(7, 8, device=device, requires_grad=True)
            attn = OTGuidedAttention(dim=8, d_attn=8, dropout=0.).to(device)
            idx = torch.randint(4, (7, 2), device=device)
            weights = torch.ones(7, 2, device=device)
            conf = torch.ones(7, device=device)
            with NoDensePairs(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                n = pool(h, self.edges.to(device), self.weights.to(device))
                value = .75 * h + .25 * n
                self.assertEqual((value.dtype, value.device), (h.dtype, h.device))
                result = attn(a, h, idx, weights, conf, target_value=value,
                              source_chunk_size=2, checkpoint_attention=True)
                loss = result.float().square().sum()
            loss.backward()
            self.assertTrue(torch.isfinite(h.grad).all())
            self.assertGreater(h.grad.abs().sum().item(), 0.)

    def test_model_bidirectional_multisection_and_zero_bypass(self):
        features, spatial = make_inputs(4)
        config = make_config('cpu')
        config['graphsage']['post_ot_graphsage_scale'] = .5
        model = StageMultiModalModel(config=config, feature_dict=features)
        initialize_prior(model, features)
        prior = copy.deepcopy(model.ot_prior)
        with patch('model.stage_model.spatial_pool_self_excluded', side_effect=AssertionError('rho=0 pooled')):
            baseline = model(features, spatial, section_order=['A', 'B', 'C'])
        old_config = copy.deepcopy(config)
        del old_config['ot_attention']['interaction_neighbor_weight']
        legacy = StageMultiModalModel(config=old_config, feature_dict=features)
        legacy.load_state_dict(model.state_dict()); legacy.ot_prior = copy.deepcopy(prior)
        legacy.ot_attention.__class__ = LegacyAttention
        old = legacy(features, spatial, section_order=['A', 'B', 'C'])
        for section in features:
            self.assertTrue(torch.equal(baseline['final_embeddings'][section], old['final_embeddings'][section]))
        model.interaction_neighbor_weight = .25
        seen = []
        original = model.ot_attention.compute_update_only
        def capture(**kwargs):
            seen.append(kwargs)
            return original(**kwargs)
        with patch.object(model.ot_attention, 'compute_update_only', side_effect=capture):
            outputs = model(features, spatial, section_order=['A', 'B', 'C'],
                            ot_attention_source_chunk_size=7, checkpoint_ot_attention=True,
                            checkpoint_graph_encoder=True)
        self.assertEqual(len(seen), 4)
        for (src, dst), call in zip(prior, seen):
            h = outputs['graphsage_embeddings'][dst]
            graph = outputs['spatial_graph_dict'][dst]
            expected = .75 * h + .25 * pool(h, graph['edge_index'], graph['edge_weight'])
            torch.testing.assert_close(call['target_value'], expected)
            self.assertIs(call['target_h'], h)
            self.assertIs(call['source_h'], outputs['graphsage_embeddings'][src])
            self.assertTrue(call['target_value'].requires_grad)
            for key in ['topk_idx', 'topk_weight', 'confidence']:
                self.assertTrue(torch.equal(model.ot_prior[(src, dst)][key], prior[(src, dst)][key]))
        for section in features:
            outputs['graphsage_embeddings'][section].retain_grad()
        sum(v.square().sum() for v in outputs['final_embeddings'].values()).backward()
        for section in features:
            self.assertGreater(outputs['graphsage_embeddings'][section].grad.abs().sum().item(), 0.)
        embeddings, contexts, diagnostics = model.prepare_ot_prior_refresh(outputs)
        for section in features:
            self.assertTrue(torch.equal(embeddings[section], outputs['ot_embeddings'][section]))
            self.assertFalse(contexts[section].requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_model_cuda_amp_training(self):
        features, spatial = make_inputs(8)
        config = make_config('cuda')
        config['ot_attention']['interaction_neighbor_weight'] = .25
        config['graphsage']['post_ot_graphsage_scale'] = .5
        model = StageMultiModalModel(config=config, feature_dict=features).cuda()
        initialize_prior(model, features)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(features, spatial, section_order=['A', 'B', 'C'],
                            ot_attention_source_chunk_size=7, checkpoint_ot_attention=True,
                            checkpoint_graph_encoder=True, checkpoint_encoder_fusion=True,
                            checkpoint_decoder_chunks=True, decoder_chunk_size=8,
                            training_loss_only=True)
            loss = outputs['losses']['total_loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.ot_attention.W_V.weight.grad).all())
        self.assertGreater(model.ot_attention.W_V.weight.grad.abs().sum().item(), 0.)

    def test_config_precedence_and_validation(self):
        key = 'interaction_neighbor_weight'
        layer = lambda rho: {'ot_attention': {key: rho}}
        self.assertEqual(resolve_model_config()['ot_attention'][key], 0.)
        preset = resolve_model_config(model_config=layer(.1))
        for cli, expected in [(None, .25), (0., 0.), (.5, .5)]:
            cfg = resolve_model_config(base=preset, input_config=layer(.25), explicit_overrides=layer(cli))
            self.assertEqual(json.loads(serialize_config(cfg))['ot_attention'][key], expected)
        for bad in [-.1, 1.1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): resolve_model_config(model_config=layer(bad))
        from scripts.run_mousebrain import parse_args, resolve_run_config
        for cli, expected in [([], .25), (['--interaction_neighbor_weight', '0'], 0.)]:
            args = parse_args(['--config', 'unused.json'] + cli)
            cfg, resolved = resolve_run_config({'model': layer(.25)}, args)
            self.assertEqual(cfg['ot_attention'][key], expected)
            self.assertEqual(resolved.interaction_neighbor_weight, expected)
        from scripts.paired_cli import parse_args as paired_args, resolve_run_config as paired_config
        audit = paired_config(paired_args(['--interaction_neighbor_weight', '.25']))
        self.assertEqual(audit['model_config']['ot_attention'][key], .25)


if __name__ == '__main__':
    unittest.main()
