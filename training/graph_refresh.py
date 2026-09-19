"""Evaluation passes before scheduled UOT refresh; no experiment orchestration."""
from __future__ import annotations


def prepare_refresh_outputs(model, forward, *, epoch):
    """Called after optimizer.step, under eval/no_grad, without autocast.

    Disabled feature graphs retain the single-pass path. Enabled graphs need
    a fused pass, graph replacement, and a fresh pass using the new graph.
    """
    import torch

    if model.training or torch.is_grad_enabled():
        raise RuntimeError("OT graph refresh requires eval mode and no_grad.")
    outputs = forward()
    graph = model.feature_graph
    if not graph.enabled:
        return outputs, None
    was_empty = not graph.edges
    sections = graph.refresh(outputs["fused_embeddings"])
    del outputs
    outputs = forward()
    return outputs, {
        "epoch": int(epoch), "feature_graph_generation": graph.generation,
        "model_stage": "after_optimizer_step", "fused_embedding_mode": "eval/no_grad",
        "refresh_passes_before_uot": 2,
        "uot_refresh_embeddings_from_updated_feature_graph": True,
        "cache_empty_before_refresh": was_empty, "sections": sections,
    }
