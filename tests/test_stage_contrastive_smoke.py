"""CPU smoke tests for the loss-only projection branch in the stage model."""

from __future__ import annotations

import unittest

import torch

from model.stage_model import (
    StageMultiModalModel,
    _make_contrastive_generator,
    _select_paired_contrastive_indices,
)


def _smoke_config() -> dict:
    return {
        "training": {"device": "cpu"},
        "model": {"latent_dim": 8},
        "encoder": {
            "hidden_dims": [16],
            "output_dim": 8,
            "dropout": 0.0,
        },
        "contrastive": {
            "method": "symmetric_infonce_vicreg",
            "projection_hidden_dim": 8,
            "projection_dim": 4,
            "projection_dropout": 0.0,
            "temperature": 0.2,
            "normalize_for_infonce": True,
            "negative_mode": "in_batch",
            "max_batch_size": 4,
            "lambda_var": 1.0,
            "lambda_cov": 0.04,
            "pair_reduction": "mean",
            "section_reduction": "mean",
        },
        "fusion": {
            "hidden_dims": [16],
            "output_dim": 8,
            "dropout": 0.0,
        },
        "graph": {"knn_neighbors_spatial": 1},
        "graphsage": {
            "enabled": False,
            "input_dim": 8,
            "output_dim": 8,
            "dropout": 0.0,
        },
        "uot": {"enabled": False},
        "ot_attention": {
            "enabled": False,
            "d_attn": 8,
            "dropout": 0.0,
        },
        "decoder": {
            "enabled": False,
            "hidden_dim": 8,
            "dropout": 0.0,
        },
        "reconstruction": {"enabled": False},
        "loss": {
            "lambda_contrast": 1.0,
            "lambda_reconstruction": 0.0,
        },
    }


def _section(n_spots: int = 8) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    features = {
        "HE": torch.randn(n_spots, 6),
        "RNA": torch.randn(n_spots, 10),
    }
    coordinates = torch.stack(
        [
            torch.arange(n_spots, dtype=torch.float32),
            torch.arange(n_spots, dtype=torch.float32) % 2,
        ],
        dim=1,
    )
    return features, coordinates


class StageContrastiveSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)

    def assert_module_has_finite_nonzero_gradient(self, module: torch.nn.Module) -> None:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_epoch_section_sampling_is_reproducible(self) -> None:
        device = torch.device("cpu")
        first = _select_paired_contrastive_indices(
            n_spots=64,
            batch_size=16,
            device=device,
            training=True,
            generator=_make_contrastive_generator(device, 3, 5, "SectionA"),
        )
        repeated = _select_paired_contrastive_indices(
            n_spots=64,
            batch_size=16,
            device=device,
            training=True,
            generator=_make_contrastive_generator(device, 3, 5, "SectionA"),
        )
        next_epoch = _select_paired_contrastive_indices(
            n_spots=64,
            batch_size=16,
            device=device,
            training=True,
            generator=_make_contrastive_generator(device, 3, 6, "SectionA"),
        )

        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, next_epoch))
        self.assertEqual(int(first.unique().numel()), 16)

    def test_forward_backward_and_sample_before_projection(self) -> None:
        section_features, coordinates = _section()
        feature_dict = {"s0": section_features}
        spatial_loc_dict = {"s0": coordinates}
        model = StageMultiModalModel(config=_smoke_config(), feature_dict=feature_dict)
        model.train()

        seen_projection_rows: dict[str, list[int]] = {"HE": [], "RNA": []}
        hooks = []
        for modality in seen_projection_rows:
            hooks.append(
                model.contrastive_projection_heads[modality].register_forward_pre_hook(
                    lambda _module, args, modality=modality: seen_projection_rows[
                        modality
                    ].append(int(args[0].shape[0]))
                )
            )

        try:
            outputs = model(
                feature_dict,
                spatial_loc_dict,
                section_order=["s0"],
                checkpoint_encoder_fusion=True,
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(seen_projection_rows, {"HE": [4], "RNA": [4]})
        self.assertEqual(tuple(outputs["final_embeddings"]["s0"].shape), (8, 8))
        self.assertEqual(int(outputs["contrastive_metrics"]["contrastive_batch_size"]), 4)
        self.assertEqual(int(outputs["contrastive_metrics"]["contrastive_num_negatives"]), 3)

        for loss in outputs["losses"].values():
            self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(outputs["losses"]["crossview_loss"]), 0.0)
        self.assertEqual(float(outputs["losses"]["reconstruction_loss"]), 0.0)
        expected_crossview = (
            outputs["losses"]["contrastive_infonce_loss"]
            + model.config["contrastive"]["lambda_var"]
            * outputs["losses"]["contrastive_variance_loss"]
            + model.config["contrastive"]["lambda_cov"]
            * outputs["losses"]["contrastive_covariance_loss"]
        )
        torch.testing.assert_close(outputs["losses"]["crossview_loss"], expected_crossview)
        torch.testing.assert_close(
            outputs["losses"]["total_loss"],
            outputs["losses"]["weighted_crossview_loss"],
        )

        # Details may retain scalar loss nodes, but must never retain a BxB
        # logits matrix or a BxD projection tensor.
        for value in outputs["loss_details"]["crossview"]["s0"].values():
            self.assertEqual(value.ndim, 0)

        outputs["losses"]["total_loss"].backward()
        for modality in ("HE", "RNA"):
            self.assert_module_has_finite_nonzero_gradient(model.encoders[modality])
            self.assert_module_has_finite_nonzero_gradient(
                model.contrastive_projection_heads[modality]
            )

        # With reconstruction disabled, total_loss has no downstream fusion
        # term. This confirms that the new projection branch is loss-only.
        self.assertTrue(
            all(parameter.grad is None for parameter in model.fusion_modules.parameters())
        )

    def test_projection_head_does_not_change_downstream_embedding(self) -> None:
        section_features, coordinates = _section()
        feature_dict = {"s0": section_features}
        spatial_loc_dict = {"s0": coordinates}
        model = StageMultiModalModel(config=_smoke_config(), feature_dict=feature_dict)
        model.eval()

        with torch.no_grad():
            before = model(feature_dict, spatial_loc_dict, section_order=["s0"])
            embedding_before = before["final_embeddings"]["s0"].clone()
            for head in model.contrastive_projection_heads.values():
                for parameter in head.parameters():
                    parameter.zero_()
            after = model(feature_dict, spatial_loc_dict, section_order=["s0"])

        torch.testing.assert_close(
            embedding_before,
            after["final_embeddings"]["s0"],
            rtol=0,
            atol=0,
        )

    def test_projection_initialization_preserves_matched_shared_weights(self) -> None:
        section_features, _ = _section()
        feature_dict = {"s0": section_features}
        legacy_config = _smoke_config()
        legacy_config["contrastive"]["method"] = "legacy_cosie_dimension"
        new_config = _smoke_config()

        torch.manual_seed(123)
        legacy_model = StageMultiModalModel(
            config=legacy_config,
            feature_dict=feature_dict,
        )
        legacy_rng_state = torch.random.get_rng_state().clone()
        torch.manual_seed(123)
        new_model = StageMultiModalModel(
            config=new_config,
            feature_dict=feature_dict,
        )
        new_rng_state = torch.random.get_rng_state().clone()

        legacy_state = legacy_model.state_dict()
        new_state = new_model.state_dict()
        shared_keys = sorted(set(legacy_state) & set(new_state))
        self.assertTrue(shared_keys)
        self.assertTrue(torch.equal(legacy_rng_state, new_rng_state))
        for key in shared_keys:
            torch.testing.assert_close(
                legacy_state[key],
                new_state[key],
                rtol=0,
                atol=0,
                msg=lambda message, key=key: f"Shared parameter {key}: {message}",
            )

    def test_corrected_cosie_uses_no_projection_head_and_has_valid_joint(self) -> None:
        section_features, coordinates = _section()
        config = _smoke_config()
        config["contrastive"]["method"] = "corrected_cosie_dimension"
        model = StageMultiModalModel(
            config=config,
            feature_dict={"s0": section_features},
        )
        self.assertEqual(len(model.contrastive_projection_heads), 0)
        model.train()
        outputs = model(
            {"s0": section_features},
            {"s0": coordinates},
            section_order=["s0"],
            training_loss_only=True,
            return_full_outputs=False,
        )
        self.assertTrue(all(torch.isfinite(loss) for loss in outputs["losses"].values()))
        self.assertGreaterEqual(
            outputs["contrastive_metrics"]["corrected_joint_min"], 0.0
        )
        self.assertAlmostEqual(
            outputs["contrastive_metrics"]["corrected_joint_sum"], 1.0, places=6
        )
        outputs["losses"]["total_loss"].backward()
        for modality in ("HE", "RNA"):
            self.assert_module_has_finite_nonzero_gradient(model.encoders[modality])

    def test_corrected_cosie_sections_are_reduced_by_mean(self) -> None:
        section_features, coordinates = _section()
        feature_dict_two = {
            "s0": section_features,
            "s1": {name: value.clone() for name, value in section_features.items()},
        }
        config = _smoke_config()
        config["contrastive"]["method"] = "corrected_cosie_dimension"
        model = StageMultiModalModel(config=config, feature_dict=feature_dict_two)
        model.eval()
        with torch.no_grad():
            one_section = model(
                {"s0": feature_dict_two["s0"]},
                {"s0": coordinates},
                section_order=["s0"],
            )
            two_sections = model(
                feature_dict_two,
                {"s0": coordinates, "s1": coordinates.clone()},
                section_order=["s0", "s1"],
            )
        torch.testing.assert_close(
            one_section["losses"]["crossview_loss"],
            two_sections["losses"]["crossview_loss"],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_crc_style_training_loss_only_path(self) -> None:
        section_features, coordinates = _section()
        feature_dict = {"s0": section_features}
        spatial_loc_dict = {"s0": coordinates}
        model = StageMultiModalModel(config=_smoke_config(), feature_dict=feature_dict)
        model.train()

        outputs = model(
            feature_dict,
            spatial_loc_dict,
            section_order=["s0"],
            training_loss_only=True,
            return_full_outputs=False,
            checkpoint_encoder_fusion=True,
        )

        self.assertNotIn("final_embeddings", outputs)
        self.assertEqual(outputs["contrastive_metrics"]["contrastive_batch_size"], 4.0)
        self.assertTrue(all(torch.isfinite(loss) for loss in outputs["losses"].values()))
        outputs["losses"]["total_loss"].backward()
        for modality in ("HE", "RNA"):
            self.assert_module_has_finite_nonzero_gradient(model.encoders[modality])
            self.assert_module_has_finite_nonzero_gradient(
                model.contrastive_projection_heads[modality]
            )

    def test_mousebrain_style_three_modality_path(self) -> None:
        section_features, coordinates = _section()
        section_features["Metabolite"] = torch.randn(8, 5)
        feature_dict = {"s0": section_features}
        spatial_loc_dict = {"s0": coordinates}
        model = StageMultiModalModel(config=_smoke_config(), feature_dict=feature_dict)
        model.train()

        outputs = model(feature_dict, spatial_loc_dict, section_order=["s0"])
        pair_keys = [
            key
            for key in outputs["loss_details"]["crossview"]["s0"]
            if key.endswith("/infonce")
        ]
        self.assertEqual(len(pair_keys), 3)
        self.assertTrue(bool(torch.isfinite(outputs["losses"]["total_loss"])))
        outputs["losses"]["total_loss"].backward()
        for modality in ("HE", "RNA", "Metabolite"):
            self.assert_module_has_finite_nonzero_gradient(model.encoders[modality])
            self.assert_module_has_finite_nonzero_gradient(
                model.contrastive_projection_heads[modality]
            )

    def test_identical_sections_are_reduced_by_mean(self) -> None:
        section_features, coordinates = _section()
        feature_dict_two = {
            "s0": section_features,
            "s1": {name: value.clone() for name, value in section_features.items()},
        }
        spatial_two = {"s0": coordinates, "s1": coordinates.clone()}
        model = StageMultiModalModel(config=_smoke_config(), feature_dict=feature_dict_two)
        model.eval()

        feature_dict_one = {"s0": feature_dict_two["s0"]}
        spatial_one = {"s0": spatial_two["s0"]}
        with torch.no_grad():
            one_section = model(feature_dict_one, spatial_one, section_order=["s0"])
            two_sections = model(
                feature_dict_two,
                spatial_two,
                section_order=["s0", "s1"],
            )

        torch.testing.assert_close(
            one_section["losses"]["crossview_loss"],
            two_sections["losses"]["crossview_loss"],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_legacy_baseline_preserves_pre_v2_section_sum(self) -> None:
        section_features, coordinates = _section()
        feature_dict_two = {
            "s0": section_features,
            "s1": {name: value.clone() for name, value in section_features.items()},
        }
        spatial_two = {"s0": coordinates, "s1": coordinates.clone()}
        config = _smoke_config()
        config["contrastive"]["method"] = "legacy_cosie_dimension"
        model = StageMultiModalModel(config=config, feature_dict=feature_dict_two)
        model.eval()

        with torch.no_grad():
            one_section = model(
                {"s0": feature_dict_two["s0"]},
                {"s0": spatial_two["s0"]},
                section_order=["s0"],
            )
            two_sections = model(
                feature_dict_two,
                spatial_two,
                section_order=["s0", "s1"],
            )

        torch.testing.assert_close(
            two_sections["losses"]["crossview_loss"],
            2.0 * one_section["losses"]["crossview_loss"],
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
