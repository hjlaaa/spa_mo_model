"""Unit tests for symmetric InfoNCE and VICReg cross-view losses."""

from __future__ import annotations

import unittest

import torch

from model.loss import (
    compute_corrected_cosie_joint,
    compute_pairwise_corrected_cosie_crossview_loss,
    compute_pairwise_crossview_loss,
    corrected_cosie_dimension_loss,
    symmetric_infonce_loss,
    vicreg_regularization,
)


class CorrectedCOSIEDimensionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)

    def test_joint_is_nonnegative_normalized_and_symmetric(self) -> None:
        view_a = torch.randn(31, 9)
        view_b = torch.randn(31, 9)
        joint_ab = compute_corrected_cosie_joint(view_a, view_b)
        joint_ba = compute_corrected_cosie_joint(view_b, view_a)

        self.assertGreaterEqual(float(joint_ab.min()), 0.0)
        torch.testing.assert_close(
            joint_ab.sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(joint_ab, joint_ab.transpose(0, 1))
        torch.testing.assert_close(joint_ab, joint_ba)

    def test_negative_inputs_and_gradients_are_finite(self) -> None:
        view_a = (-torch.rand(23, 7)).requires_grad_(True)
        view_b = torch.zeros(23, 7, requires_grad=True)
        loss, diagnostics = corrected_cosie_dimension_loss(view_a, view_b)

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(diagnostics["corrected_joint_min"]), 0.0)
        torch.testing.assert_close(
            diagnostics["corrected_joint_sum"],
            torch.tensor(1.0),
            rtol=1e-6,
            atol=1e-7,
        )
        loss.backward()
        for gradient in (view_a.grad, view_b.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_pair_mean_is_invariant_to_duplicating_identical_modalities(self) -> None:
        view = torch.randn(19, 6)
        two_view, _ = compute_pairwise_corrected_cosie_crossview_loss(
            {"HE": view, "RNA": view.clone()}
        )
        three_view, _ = compute_pairwise_corrected_cosie_crossview_loss(
            {"HE": view, "RNA": view.clone(), "Metabolite": view.clone()}
        )
        torch.testing.assert_close(two_view, three_view, rtol=1e-6, atol=1e-7)

    def test_cpu_bfloat16_autocast_is_finite(self) -> None:
        view_a = torch.randn(24, 8, requires_grad=True)
        view_b = torch.randn(24, 8, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss, _ = corrected_cosie_dimension_loss(view_a, view_b)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(view_a.grad).all()))
        self.assertTrue(bool(torch.isfinite(view_b.grad).all()))

    def test_invalid_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compute_corrected_cosie_joint(torch.randn(3, 4), torch.randn(3, 5))
        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            compute_corrected_cosie_joint(
                torch.randn(3, 4), torch.randn(3, 4), temperature=0
            )


class SymmetricInfoNCETest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_matched_pairs_score_better_than_shuffled_pairs(self) -> None:
        view_a = torch.randn(64, 32)
        view_b = view_a + 0.02 * torch.randn_like(view_a)

        matched = symmetric_infonce_loss(view_a, view_b, temperature=0.2)
        shuffled = symmetric_infonce_loss(
            view_a,
            view_b.roll(shifts=1, dims=0),
            temperature=0.2,
        )

        self.assertLess(float(matched), float(shuffled))

    def test_loss_is_symmetric(self) -> None:
        view_a = torch.randn(23, 11)
        view_b = torch.randn(23, 11)

        loss_ab = symmetric_infonce_loss(view_a, view_b, temperature=0.37)
        loss_ba = symmetric_infonce_loss(view_b, view_a, temperature=0.37)

        torch.testing.assert_close(loss_ab, loss_ba, rtol=1e-6, atol=1e-7)

    def test_negative_and_zero_inputs_are_finite_with_finite_gradients(self) -> None:
        view_a = (-torch.rand(19, 13)).requires_grad_(True)
        view_b = torch.zeros(19, 13, requires_grad=True)

        loss = symmetric_infonce_loss(view_a, view_b)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(loss), 0.0)
        loss.backward()

        for gradient in (view_a.grad, view_b.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertLess(float(gradient.abs().max()), 1e4)

    def test_cpu_bfloat16_autocast_stays_finite(self) -> None:
        view_a = torch.randn(24, 16, requires_grad=True)
        view_b = torch.randn(24, 16, requires_grad=True)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = symmetric_infonce_loss(view_a, view_b)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(view_a.grad).all()))
        self.assertTrue(bool(torch.isfinite(view_b.grad).all()))

    def test_invalid_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two paired spots"):
            symmetric_infonce_loss(torch.randn(1, 4), torch.randn(1, 4))
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            symmetric_infonce_loss(torch.randn(4, 3), torch.randn(4, 5))
        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            symmetric_infonce_loss(torch.randn(4, 3), torch.randn(4, 3), temperature=0)
        with self.assertRaises(NotImplementedError):
            symmetric_infonce_loss(
                torch.randn(4, 3),
                torch.randn(4, 3),
                negative_mode="queue",
            )


class VICRegTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_collapsed_projection_has_larger_variance_penalty(self) -> None:
        collapsed = torch.zeros(256, 8)
        dispersed = 1.5 * torch.randn(256, 8)

        collapsed_variance, _ = vicreg_regularization([collapsed])
        dispersed_variance, _ = vicreg_regularization([dispersed])

        self.assertGreater(float(collapsed_variance), float(dispersed_variance))

    def test_correlated_dimensions_have_larger_covariance_penalty(self) -> None:
        independent = torch.randn(512, 4)
        correlated = torch.randn(512, 4)
        correlated[:, 1] = correlated[:, 0]

        _, independent_covariance = vicreg_regularization([independent])
        _, correlated_covariance = vicreg_regularization([correlated])

        self.assertGreater(float(correlated_covariance), float(independent_covariance))

    def test_three_modalities_use_pair_and_modality_means(self) -> None:
        projected = {
            "HE": 0.5 * torch.randn(32, 10),
            "RNA": 1.2 * torch.randn(32, 10),
            "Metabolite": 2.0 * torch.randn(32, 10),
        }
        config = {
            "method": "symmetric_infonce_vicreg",
            "temperature": 0.2,
            "lambda_var": 1.0,
            "lambda_cov": 0.04,
        }

        three_view_total, three_view_detail = compute_pairwise_crossview_loss(
            projected, config
        )

        pair_keys = [key for key in three_view_detail if key.endswith("/infonce")]
        self.assertEqual(len(pair_keys), 3)
        manual_infonce = torch.stack(
            [
                symmetric_infonce_loss(projected["HE"], projected["RNA"]),
                symmetric_infonce_loss(projected["HE"], projected["Metabolite"]),
                symmetric_infonce_loss(projected["RNA"], projected["Metabolite"]),
            ]
        ).mean()
        manual_variance, manual_covariance = vicreg_regularization(projected)
        torch.testing.assert_close(three_view_detail["infonce"], manual_infonce)
        torch.testing.assert_close(three_view_detail["variance"], manual_variance)
        torch.testing.assert_close(three_view_detail["covariance"], manual_covariance)
        expected = (
            manual_infonce
            + config["lambda_var"] * manual_variance
            + config["lambda_cov"] * manual_covariance
        )
        torch.testing.assert_close(three_view_total, expected)
        self.assertEqual(int(three_view_detail["contrastive_num_negatives"]), 31)

    def test_composite_loss_backward_is_finite(self) -> None:
        projected = {
            "HE": torch.randn(17, 7, requires_grad=True),
            "RNA": torch.randn(17, 7, requires_grad=True),
            "Metabolite": torch.randn(17, 7, requires_grad=True),
        }
        total, detail = compute_pairwise_crossview_loss(
            projected,
            {
                "method": "symmetric_infonce_vicreg",
                "temperature": 0.2,
                "lambda_var": 1.0,
                "lambda_cov": 0.04,
            },
        )

        self.assertTrue(bool(torch.isfinite(total)))
        self.assertTrue(bool(torch.isfinite(detail["crossmodal_top1_accuracy"])))
        total.backward()
        for view in projected.values():
            self.assertIsNotNone(view.grad)
            self.assertTrue(bool(torch.isfinite(view.grad).all()))


if __name__ == "__main__":
    unittest.main()
