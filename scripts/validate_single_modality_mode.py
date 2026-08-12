#!/usr/bin/env python3
"""Fast CPU regression checks for the opt-in single-modality model path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_model_config
from model.stage_model import StageMultiModalModel


def model_config(single_modality: str | None = None) -> dict:
    config = get_default_model_config()
    config["training"]["device"] = "cpu"
    config["uot"]["enabled"] = False
    config["ot_attention"]["enabled"] = False
    config["model"]["single_modality_mode"] = {
        "enabled": single_modality is not None,
        "modality": single_modality,
    }
    return config


def main() -> None:
    coords = {
        "s1": np.column_stack((np.arange(8), np.zeros(8))).astype(np.float32),
        "s2": np.column_stack((np.arange(7), np.ones(7))).astype(np.float32),
    }
    for modality in ("RNA", "Protein", "HE"):
        rng = np.random.default_rng(42)
        features = {
            "s1": {modality: rng.normal(size=(8, 6)).astype(np.float32)},
            "s2": {modality: rng.normal(size=(7, 6)).astype(np.float32)},
        }
        model = StageMultiModalModel(model_config(modality), features)
        outputs = model(features, coords, section_order=["s1", "s2"], training_loss_only=True)
        outputs["losses"]["total_loss"].backward()
        assert outputs["mode"] == {
            "single_modality": True,
            "modalities": [modality],
            "contrastive_skipped": True,
            "fusion": "identity",
        }
        assert outputs["losses"]["crossview_loss"].item() == 0.0
        assert any(parameter.grad is not None for parameter in model.parameters())
        print(f"{modality}-only: PASS")

    single = {
        "s1": {"RNA": np.ones((8, 6), dtype=np.float32)},
        "s2": {"RNA": np.ones((7, 6), dtype=np.float32)},
    }
    try:
        StageMultiModalModel(model_config(), single)
    except ValueError as error:
        assert "single-modality mode is disabled" in str(error)
    else:
        raise AssertionError("The backward-compatible default unexpectedly accepted one modality.")
    print("default single-modality guard: PASS")

    rng = np.random.default_rng(7)
    multimodal = {
        "s1": {
            "RNA": rng.normal(size=(8, 6)).astype(np.float32),
            "Protein": rng.normal(size=(8, 4)).astype(np.float32),
        },
        "s2": {
            "RNA": rng.normal(size=(7, 6)).astype(np.float32),
            "Protein": rng.normal(size=(7, 4)).astype(np.float32),
        },
    }
    model = StageMultiModalModel(model_config(), multimodal)
    outputs = model(multimodal, coords, section_order=["s1", "s2"], training_loss_only=True)
    outputs["losses"]["total_loss"].backward()
    assert outputs["mode"]["single_modality"] is False
    assert outputs["losses"]["crossview_loss"].item() != 0.0
    print("RNA+Protein regression: PASS")


if __name__ == "__main__":
    main()
