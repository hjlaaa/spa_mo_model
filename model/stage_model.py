"""Stage-level multimodal model after COSIE-style preprocessing."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .configure import get_default_model_config
from .linkage_construction import (
    compute_initial_multimodal_uot_prior,
    update_uot_prior_from_embeddings,
)
from .loss import (
    compute_pairwise_cosie_crossview_loss,
)
from .model_component import (
    FusionMLP,
    ModalityDecoder,
    ModalityMLPEncoder,
    OTGuidedAttention,
    WeightedResidualGraphSAGE,
)
from .sparse_uot import (
    compute_initial_bidirectional_candidate_sparse_uot_prior,
    compute_initial_candidate_sparse_uot_prior,
    update_bidirectional_candidate_sparse_uot_prior_from_embeddings,
    update_candidate_sparse_uot_prior_from_embeddings,
)
from .utils import compute_spatial_knn_graph_with_weights


def _recursive_update(base: dict[str, Any], updates: Mapping[str, Any] | None) -> dict[str, Any]:
    if updates is None:
        return base
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


ForwardMemoryRecorder = Callable[[str, Any], None]


def _record_forward_memory(
    recorder: ForwardMemoryRecorder | None,
    stage: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    if recorder is not None:
        recorder(stage, extra)


def _checkpoint_module_forward(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    def run_module(*args: torch.Tensor) -> torch.Tensor:
        return module(*args)

    return torch_checkpoint(
        run_module,
        *inputs,
        use_reentrant=False,
        preserve_rng_state=True,
    )


def should_update_ot(epoch: int, update_interval: int = 20) -> bool:
    """Return True when the cached UOT prior should be refreshed.

    Recommended training pattern:

    ``model.initialize_ot_prior(feature_dict, section_order)``

    ``for epoch in range(1, epochs + 1):``

    ``    outputs = model(feature_dict, spatial_loc_dict, section_order=section_order, epoch=epoch)``

    ``    loss = outputs["losses"]["total_loss"]``

    ``    optimizer.zero_grad(); loss.backward(); optimizer.step()``

    ``    if should_update_ot(epoch, 20):``

    ``        with torch.no_grad():``

    ``            eval_outputs = model(feature_dict, spatial_loc_dict, section_order=section_order, epoch=epoch)``

    ``            model.update_ot_prior(eval_outputs["final_embeddings"], section_order)``
    """

    return epoch > 0 and epoch % update_interval == 0


class StageMultiModalModel(nn.Module):
    """V2 stage model built on the V1 COSIE-style multimodal backbone.

    V1 is preserved:
    ``feature_dict/spatial_loc_dict -> modality MLP encoders -> 128-d latents
    -> COSIE cross-view loss -> FusionMLP -> 128-d fused embeddings``.

    V2 appends weighted residual GraphSAGE, adjacent-stage UOT-guided
    attention, and modality decoders. UOT is cached as a prior and does not
    participate in backpropagation.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        feature_dict: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        super().__init__()
        self.config = _recursive_update(get_default_model_config(), config)
        self.latent_dim = int(self.config["model"]["latent_dim"])
        self.canonical_modality_order = tuple(self.config["model"]["modalities_supported"])
        self.valid_modality_sets = [
            tuple(
                modality
                for modality in self.canonical_modality_order
                if modality in set(modalities)
            )
            for modalities in self.config["model"]["valid_modality_sets"]
        ]
        self.input_dims: dict[str, int] = {}
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.ot_prior: dict[tuple[str, str], dict[str, Any]] | None = None
        self._spatial_graph_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}

        self.fusion_modules = nn.ModuleDict()
        for modality_order in self.valid_modality_sets:
            key = self._combo_key(modality_order)
            self.fusion_modules[key] = FusionMLP(
                modality_order=modality_order,
                latent_dim=self.latent_dim,
                hidden_dims=self.config["fusion"]["hidden_dims"],
                output_dim=self.config["fusion"]["output_dim"],
                activation=self.config["fusion"]["activation"],
                dropout=float(self.config["fusion"]["dropout"]),
                norm=self.config["fusion"]["norm"],
                residual=self.config["fusion"]["residual"],
            )

        graph_cfg = self.config["graphsage"]
        self.graphsage = WeightedResidualGraphSAGE(
            input_dim=int(graph_cfg["input_dim"]),
            output_dim=int(graph_cfg["output_dim"]),
            dropout=float(graph_cfg["dropout"]),
            activation=graph_cfg["activation"],
            norm=graph_cfg["norm"],
            residual=bool(graph_cfg["residual"]),
            edge_batch_size=graph_cfg.get("edge_batch_size", 200000),
        )

        attn_cfg = self.config["ot_attention"]
        self.ot_attention = OTGuidedAttention(
            dim=self.latent_dim,
            d_attn=int(attn_cfg["d_attn"]),
            beta=float(attn_cfg["beta"]),
            beta_warmup=bool(attn_cfg["beta_warmup"]),
            beta_schedule=attn_cfg["beta_schedule"],
            dropout=float(attn_cfg["dropout"]),
            gate=attn_cfg["gate"],
            use_confidence=bool(attn_cfg["use_confidence"]),
            residual=bool(attn_cfg["residual"]),
            norm=attn_cfg["norm"],
            delta=float(attn_cfg["delta"]),
        )

        if feature_dict is not None:
            self.initialize_from_feature_dict(feature_dict)

    @staticmethod
    def _combo_key(modality_order: tuple[str, ...] | list[str]) -> str:
        return "__".join(modality_order)

    def _resolve_section_order(
        self,
        feature_dict: Mapping[str, Mapping[str, Any]],
        section_order: Sequence[str] | None,
    ) -> list[str]:
        if section_order is None:
            return sorted(feature_dict.keys())
        missing = [section for section in section_order if section not in feature_dict]
        if missing:
            raise KeyError(f"section_order contains sections missing from feature_dict: {missing}")
        extras = [section for section in feature_dict.keys() if section not in section_order]
        if extras:
            raise KeyError(f"feature_dict contains sections missing from section_order: {extras}")
        return list(section_order)

    def _resolve_modality_order(self, modalities: Mapping[str, Any] | set[str]) -> tuple[str, ...]:
        if isinstance(modalities, Mapping):
            present = {modality for modality, value in modalities.items() if value is not None}
        else:
            present = set(modalities)
        unknown = sorted(present - set(self.canonical_modality_order))
        if unknown:
            raise ValueError(f"Unsupported modalities present in section: {unknown}.")
        if len(present) < 2:
            raise ValueError(
                f"Each section must contain at least two real observed modalities; got {sorted(present)}."
            )

        ordered_present = tuple(
            modality for modality in self.canonical_modality_order if modality in present
        )
        for valid_set in self.valid_modality_sets:
            if ordered_present == valid_set:
                return ordered_present
        valid_text = ", ".join(self._combo_key(combo) for combo in self.valid_modality_sets)
        raise ValueError(
            f"Each section must contain exactly one supported two- or three-modality set. "
            f"Got {list(ordered_present)}; supported sets are: {valid_text}."
            )

    @staticmethod
    def _looks_like_section_order(value: Any) -> bool:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and all(isinstance(section, str) for section in value)
        )

    def _make_encoder(self, input_dim: int) -> ModalityMLPEncoder:
        encoder_cfg = self.config["encoder"]
        return ModalityMLPEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_cfg["hidden_dims"],
            output_dim=encoder_cfg["output_dim"],
            activation=encoder_cfg["activation"],
            dropout=float(encoder_cfg["dropout"]),
            norm=encoder_cfg["norm"],
            l2_normalize_output=bool(encoder_cfg["l2_normalize_output"]),
        )

    def _make_decoder(self, output_dim: int) -> ModalityDecoder:
        decoder_cfg = self.config["decoder"]
        return ModalityDecoder(
            input_dim=self.latent_dim,
            hidden_dim=int(decoder_cfg["hidden_dim"]),
            output_dim=output_dim,
            activation=decoder_cfg["activation"],
            dropout=float(decoder_cfg["dropout"]),
        )

    def initialize_from_feature_dict(
        self,
        feature_dict: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Create modality-specific encoders and decoders from feature shapes."""

        expected_modality_order: tuple[str, ...] | None = None
        for section, modalities in feature_dict.items():
            modality_order = self._resolve_modality_order(modalities)
            if expected_modality_order is None:
                expected_modality_order = modality_order
            elif modality_order != expected_modality_order:
                raise ValueError(
                    "All sections in one model run must use the same observed modality set; "
                    f"{section} has {list(modality_order)} but expected {list(expected_modality_order)}."
                )

            for modality in modality_order:
                features = modalities[modality]
                if features is None:
                    raise ValueError(
                        f"Missing modality {modality} in {section}; this stage "
                        "expects complete observed modalities for a supported set."
                    )
                if not hasattr(features, "shape") or len(features.shape) != 2:
                    raise ValueError(
                        f"feature_dict[{section!r}][{modality!r}] must be 2D, "
                        f"got shape {getattr(features, 'shape', None)}."
                    )
                input_dim = int(features.shape[1])
                if modality in self.input_dims and self.input_dims[modality] != input_dim:
                    raise ValueError(
                        f"Modality {modality} has inconsistent input dimensions: "
                        f"{self.input_dims[modality]} vs {input_dim}."
                    )
                if modality not in self.encoders:
                    self.input_dims[modality] = input_dim
                    self.encoders[modality] = self._make_encoder(input_dim)
                if modality not in self.decoders:
                    self.decoders[modality] = self._make_decoder(input_dim)

    def _select_device(self, feature_dict: Mapping[str, Mapping[str, Any]]) -> torch.device:
        for modalities in feature_dict.values():
            for features in modalities.values():
                if isinstance(features, torch.Tensor) and features.is_cuda:
                    return features.device

        configured = self.config["training"].get("device", "cpu")
        if configured == "cuda" and not torch.cuda.is_available():
            configured = "cpu"
        return torch.device(configured)

    @staticmethod
    def _as_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def clear_spatial_graph_cache(self) -> None:
        """Clear cached CPU spatial KNN graphs."""

        self._spatial_graph_cache.clear()

    def _get_spatial_graph(
        self,
        section: str,
        spatial_coords: Any,
        graph_cfg: Mapping[str, Any],
        graph_sage_cfg: Mapping[str, Any],
        device: torch.device,
        cache_spatial_graphs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not cache_spatial_graphs:
            return compute_spatial_knn_graph_with_weights(
                spatial_coords,
                k=int(graph_cfg["knn_neighbors_spatial"]),
                include_self_loop=True,
                undirected=True,
                delta=float(graph_sage_cfg["delta"]),
                device=device,
            )

        spatial_n = int(spatial_coords.shape[0]) if hasattr(spatial_coords, "shape") else len(spatial_coords)
        cache_key = (
            section,
            spatial_n,
            int(graph_cfg["knn_neighbors_spatial"]),
            True,
            True,
            float(graph_sage_cfg["delta"]),
        )
        if cache_key not in self._spatial_graph_cache:
            edge_index_cpu, edge_weight_cpu = compute_spatial_knn_graph_with_weights(
                spatial_coords,
                k=int(graph_cfg["knn_neighbors_spatial"]),
                include_self_loop=True,
                undirected=True,
                delta=float(graph_sage_cfg["delta"]),
                device=torch.device("cpu"),
            )
            self._spatial_graph_cache[cache_key] = (edge_index_cpu, edge_weight_cpu)

        edge_index_cpu, edge_weight_cpu = self._spatial_graph_cache[cache_key]
        return edge_index_cpu.to(device), edge_weight_cpu.to(device)

    def _decode_and_reconstruct_section(
        self,
        section: str,
        final_embedding: torch.Tensor,
        target_features: Mapping[str, torch.Tensor],
        lambda_by_modality: Mapping[str, float] | None,
        decoder_chunk_size: int | None = None,
        return_reconstructions: bool = True,
        memory_recorder: ForwardMemoryRecorder | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not target_features:
            zero = torch.zeros((), device=final_embedding.device, dtype=final_embedding.dtype)
            return zero, {}, {}

        weights = lambda_by_modality or {}
        chunk_size = int(decoder_chunk_size or 0)
        total_loss = torch.zeros((), device=final_embedding.device, dtype=torch.float32)
        detail: dict[str, torch.Tensor] = {}
        reconstructions: dict[str, torch.Tensor] = {}

        for modality, target in target_features.items():
            if modality not in self.decoders:
                raise KeyError(f"No decoder initialized for modality {modality}.")
            target = target.to(device=final_embedding.device, dtype=final_embedding.dtype)

            if chunk_size <= 0:
                recon = self.decoders[modality](final_embedding)
                if recon.shape != target.shape:
                    raise ValueError(
                        f"Reconstruction shape mismatch for {section}/{modality}: "
                        f"recon={tuple(recon.shape)}, target={tuple(target.shape)}."
                    )
                loss = F.mse_loss(recon.float(), target.float())
                if return_reconstructions:
                    reconstructions[modality] = recon
            else:
                sqerr = torch.zeros((), device=final_embedding.device, dtype=torch.float32)
                total_numel = 0
                recon_chunks: list[torch.Tensor] = []
                for start in range(0, int(final_embedding.shape[0]), chunk_size):
                    end = min(start + chunk_size, int(final_embedding.shape[0]))
                    recon_chunk = self.decoders[modality](final_embedding[start:end])
                    target_chunk = target[start:end]
                    if recon_chunk.shape != target_chunk.shape:
                        raise ValueError(
                            f"Reconstruction shape mismatch for {section}/{modality}: "
                            f"recon={tuple(recon_chunk.shape)}, target={tuple(target_chunk.shape)}."
                        )
                    sqerr = sqerr + F.mse_loss(
                        recon_chunk.float(),
                        target_chunk.float(),
                        reduction="sum",
                    )
                    total_numel += int(target_chunk.numel())
                    if return_reconstructions:
                        if torch.is_grad_enabled():
                            recon_chunks.append(recon_chunk)
                        else:
                            recon_chunks.append(recon_chunk.detach().cpu())
                if total_numel <= 0:
                    raise ValueError(f"Empty reconstruction target for {section}/{modality}.")
                loss = sqerr / float(total_numel)
                if return_reconstructions:
                    reconstructions[modality] = torch.cat(recon_chunks, dim=0)

            detail[modality] = loss
            total_loss = total_loss + float(weights.get(modality, 1.0)) * loss
            _record_forward_memory(
                memory_recorder,
                f"decoder_{section}_{modality}_end",
                {
                    "section": section,
                    "modality": modality,
                    "n_spots": int(final_embedding.shape[0]),
                    "output_dim": int(target.shape[1]),
                    "decoder_chunk_size": chunk_size,
                    "return_reconstructions": bool(return_reconstructions),
                },
            )

        return total_loss, detail, reconstructions

    def initialize_ot_prior(
        self,
        feature_dict: Mapping[str, Mapping[str, Any]],
        section_order: Sequence[str] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Initialize adjacent-stage UOT priors from preprocessed modality features."""

        if not self.config["uot"]["enabled"]:
            self.ot_prior = {}
            return self.ot_prior
        resolved_order = self._resolve_section_order(feature_dict, section_order)
        expected_modality_order: tuple[str, ...] | None = None
        for section in resolved_order:
            modality_order = self._resolve_modality_order(feature_dict[section])
            if expected_modality_order is None:
                expected_modality_order = modality_order
            elif modality_order != expected_modality_order:
                raise ValueError(
                    "All sections used for OT prior initialization must use the same observed modality set; "
                    f"{section} has {list(modality_order)} but expected {list(expected_modality_order)}."
                )
        uot_cfg = self.config["uot"]
        self.ot_prior = compute_initial_multimodal_uot_prior(
            feature_dict=feature_dict,
            section_order=resolved_order,
            modalities=expected_modality_order or self.config["model"]["modalities_supported"],
            epsilon_init=float(uot_cfg["epsilon_init"]),
            tau_a=float(uot_cfg["tau_a"]),
            tau_b=float(uot_cfg["tau_b"]),
            max_iter=int(uot_cfg["max_iter"]),
            tol=float(uot_cfg["tol"]),
            topk=int(uot_cfg["topk"]),
            delta=float(self.config["ot_attention"]["delta"]),
            check_every=int(uot_cfg["check_every"]),
            clip_cost_min=float(uot_cfg["clip_cost_min"]),
            clip_cost_max=float(uot_cfg["clip_cost_max"]),
            keep_dense=bool(uot_cfg.get("keep_dense", False)),
        )
        return self.ot_prior

    @torch.no_grad()
    def initialize_candidate_sparse_ot_prior(
        self,
        feature_dict: Mapping[str, Mapping[str, Any]],
        section_order: Sequence[str] | None = None,
        initial_modality_candidate_k: int = 100,
        candidate_k: int = 200,
        attention_topk: int = 10,
        candidate_backend: str = "faiss_ivf",
        faiss_nlist: int = 4096,
        faiss_nprobe: int = 64,
        faiss_device: str = "auto",
        faiss_train_sample_size: int = 100000,
        faiss_query_batch_size: int | None = 8192,
        seed: int = 42,
        epsilon: float = 0.05,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        max_iter: int = 100,
        stabilizer: float = 1e-8,
        bidirectional: bool = False,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Initialize adjacent-stage candidate-sparse UOT priors."""

        if not self.config["uot"]["enabled"]:
            self.ot_prior = {}
            return self.ot_prior
        resolved_order = self._resolve_section_order(feature_dict, section_order)
        expected_modality_order: tuple[str, ...] | None = None
        for section in resolved_order:
            modality_order = self._resolve_modality_order(feature_dict[section])
            if expected_modality_order is None:
                expected_modality_order = modality_order
            elif modality_order != expected_modality_order:
                raise ValueError(
                    "All sections used for OT prior initialization must use the same observed modality set; "
                    f"{section} has {list(modality_order)} but expected {list(expected_modality_order)}."
                )
        device = self._select_device(feature_dict)
        init_fn = (
            compute_initial_bidirectional_candidate_sparse_uot_prior
            if bidirectional
            else compute_initial_candidate_sparse_uot_prior
        )
        self.ot_prior = init_fn(
            feature_dict=feature_dict,
            section_order=resolved_order,
            modalities=expected_modality_order or self.config["model"]["modalities_supported"],
            initial_modality_candidate_k=initial_modality_candidate_k,
            candidate_k=candidate_k,
            attention_topk=attention_topk,
            candidate_backend=candidate_backend,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
            faiss_device=faiss_device,
            faiss_train_sample_size=faiss_train_sample_size,
            faiss_query_batch_size=faiss_query_batch_size,
            seed=seed,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            max_iter=max_iter,
            stabilizer=stabilizer,
            device=device,
        )
        return self.ot_prior

    @torch.no_grad()
    def update_ot_prior(
        self,
        final_embedding_dict: Mapping[str, torch.Tensor],
        section_order: Sequence[str] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Refresh adjacent-stage UOT priors from detached final embeddings."""

        if not self.config["uot"]["enabled"]:
            self.ot_prior = {}
            return self.ot_prior
        uot_cfg = self.config["uot"]
        self.ot_prior = update_uot_prior_from_embeddings(
            final_embedding_dict=final_embedding_dict,
            section_order=section_order,
            epsilon_update=float(uot_cfg["epsilon_update"]),
            tau_a=float(uot_cfg["tau_a"]),
            tau_b=float(uot_cfg["tau_b"]),
            max_iter=int(uot_cfg["max_iter"]),
            tol=float(uot_cfg["tol"]),
            topk=int(uot_cfg["topk"]),
            delta=float(self.config["ot_attention"]["delta"]),
            check_every=int(uot_cfg["check_every"]),
            clip_cost_min=float(uot_cfg["clip_cost_min"]),
            clip_cost_max=float(uot_cfg["clip_cost_max"]),
            keep_dense=bool(uot_cfg.get("keep_dense", False)),
        )
        return self.ot_prior

    @torch.no_grad()
    def update_candidate_sparse_ot_prior(
        self,
        embedding_dict: Mapping[str, torch.Tensor],
        section_order: Sequence[str] | None = None,
        candidate_k: int = 200,
        attention_topk: int = 10,
        candidate_backend: str = "faiss_ivf",
        faiss_nlist: int = 4096,
        faiss_nprobe: int = 64,
        faiss_device: str = "auto",
        faiss_train_sample_size: int = 100000,
        faiss_query_batch_size: int | None = 8192,
        seed: int = 42,
        epsilon: float = 0.05,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        max_iter: int = 100,
        stabilizer: float = 1e-8,
        candidate_source: str = "fused",
        bidirectional: bool = False,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Refresh candidate-sparse UOT priors from fused/final embeddings."""

        if not self.config["uot"]["enabled"]:
            self.ot_prior = {}
            return self.ot_prior
        device = self._select_device({section: {"embedding": value} for section, value in embedding_dict.items()})
        update_fn = (
            update_bidirectional_candidate_sparse_uot_prior_from_embeddings
            if bidirectional
            else update_candidate_sparse_uot_prior_from_embeddings
        )
        self.ot_prior = update_fn(
            embedding_dict=embedding_dict,
            section_order=section_order,
            candidate_k=candidate_k,
            attention_topk=attention_topk,
            candidate_backend=candidate_backend,
            faiss_nlist=faiss_nlist,
            faiss_nprobe=faiss_nprobe,
            faiss_device=faiss_device,
            faiss_train_sample_size=faiss_train_sample_size,
            faiss_query_batch_size=faiss_query_batch_size,
            seed=seed,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            max_iter=max_iter,
            stabilizer=stabilizer,
            device=device,
            candidate_source=candidate_source,
        )
        return self.ot_prior

    def forward(
        self,
        feature_dict: Mapping[str, Mapping[str, Any]],
        spatial_loc_dict: Mapping[str, Any],
        processed_data_dict: Any | None = None,
        section_order: Sequence[str] | None = None,
        epoch: int | None = None,
        training_loss_only: bool = False,
        return_full_outputs: bool = True,
        decoder_chunk_size: int | None = None,
        ot_attention_source_chunk_size: int | None = None,
        cache_spatial_graphs: bool = False,
        bidirectional_ot_attention: bool = False,
        checkpoint_ot_attention: bool = False,
        checkpoint_encoder_fusion: bool = False,
        memory_recorder: ForwardMemoryRecorder | None = None,
    ) -> dict[str, Any]:
        if self._looks_like_section_order(processed_data_dict):
            if section_order is None:
                section_order = processed_data_dict
                processed_data_dict = None
            elif isinstance(section_order, int) and epoch is None:
                epoch = int(section_order)
                section_order = processed_data_dict
                processed_data_dict = None

        if not self.encoders:
            self.initialize_from_feature_dict(feature_dict)

        resolved_order = self._resolve_section_order(feature_dict, section_order)
        device = self._select_device(feature_dict)
        self.to(device)

        graph_cfg = self.config["graph"]
        graph_sage_cfg = self.config["graphsage"]
        contrastive_gamma = float(self.config["contrastive"]["gamma"])
        lambda_contrast = float(self.config["loss"]["lambda_contrast"])
        lambda_reconstruction = float(self.config["loss"]["lambda_reconstruction"])
        lambda_by_modality = self.config["reconstruction"]["lambda_by_modality"]
        keep_full_outputs = bool(return_full_outputs) and not bool(training_loss_only)
        use_encoder_fusion_checkpoint = bool(
            checkpoint_encoder_fusion
            and self.training
            and torch.is_grad_enabled()
        )

        fused_embeddings: dict[str, torch.Tensor] = {}
        graphsage_embeddings: dict[str, torch.Tensor] = {}
        final_embeddings: dict[str, torch.Tensor] = {}
        latent_dict: dict[str, dict[str, torch.Tensor]] = {}
        reconstructions: dict[str, dict[str, torch.Tensor]] = {}
        target_feature_dict: dict[str, dict[str, torch.Tensor]] = {}
        spatial_graph_dict: dict[str, dict[str, torch.Tensor]] = {}
        crossview_details: dict[str, dict[str, torch.Tensor]] = {}
        reconstruction_details: dict[str, dict[str, torch.Tensor]] = {}
        messages: list[str] = [
            "processed_data_dict is accepted for pipeline compatibility but is not used by Model Stage V2."
        ]
        crossview_loss = torch.zeros((), device=device)
        reconstruction_loss = torch.zeros((), device=device)
        expected_modality_order: tuple[str, ...] | None = None
        _record_forward_memory(
            memory_recorder,
            "start",
            {
                "sections": list(resolved_order),
                "training_loss_only": bool(training_loss_only),
                "return_full_outputs": bool(return_full_outputs),
                "bidirectional_ot_attention": bool(bidirectional_ot_attention),
                "checkpoint_ot_attention": bool(checkpoint_ot_attention),
                "checkpoint_encoder_fusion": bool(checkpoint_encoder_fusion),
            },
        )

        for section in resolved_order:
            if section not in spatial_loc_dict:
                raise KeyError(f"Missing spatial coordinates for section {section}.")

            modalities = feature_dict[section]
            modality_order = self._resolve_modality_order(modalities)
            if expected_modality_order is None:
                expected_modality_order = modality_order
            elif modality_order != expected_modality_order:
                raise ValueError(
                    "All sections in one forward pass must use the same observed modality set; "
                    f"{section} has {list(modality_order)} but expected {list(expected_modality_order)}."
                )
            combo_key = self._combo_key(modality_order)

            section_features: dict[str, torch.Tensor] = {}
            n_spots: int | None = None
            for modality in modality_order:
                x_mod = self._as_float_tensor(modalities[modality], device=device)
                if x_mod.ndim != 2:
                    raise ValueError(
                        f"feature_dict[{section!r}][{modality!r}] must be 2D, got {x_mod.ndim}D."
                    )
                if n_spots is None:
                    n_spots = int(x_mod.shape[0])
                elif int(x_mod.shape[0]) != n_spots:
                    raise ValueError(
                        f"All modalities in {section} must have the same spot count; "
                        f"got {modality} with {x_mod.shape[0]} vs {n_spots}."
                    )
                section_features[modality] = x_mod
            _record_forward_memory(
                memory_recorder,
                f"section_{section}_features_end",
                {
                    "section": section,
                    "n_spots": int(n_spots or 0),
                    "modalities": list(modality_order),
                    "feature_shapes": {
                        modality: list(section_features[modality].shape)
                        for modality in modality_order
                    },
                },
            )

            spatial_coords = spatial_loc_dict[section]
            spatial_n = int(spatial_coords.shape[0]) if hasattr(spatial_coords, "shape") else len(spatial_coords)
            if n_spots is not None and spatial_n != n_spots:
                raise ValueError(
                    f"Spatial coordinates for {section} have {spatial_n} rows, "
                    f"but feature matrices have {n_spots} spots."
                )

            edge_index, edge_weight = self._get_spatial_graph(
                section=section,
                spatial_coords=spatial_coords,
                graph_cfg=graph_cfg,
                graph_sage_cfg=graph_sage_cfg,
                device=device,
                cache_spatial_graphs=cache_spatial_graphs,
            )
            _record_forward_memory(
                memory_recorder,
                f"section_{section}_spatial_graph_end",
                {
                    "section": section,
                    "edge_count": int(edge_index.shape[1]),
                    "cache_spatial_graphs": bool(cache_spatial_graphs),
                },
            )
            if keep_full_outputs:
                spatial_graph_dict[section] = {
                    "edge_index": edge_index,
                    "edge_weight": edge_weight,
                }

            section_latents: dict[str, torch.Tensor] = {}
            for modality in modality_order:
                if modality not in self.encoders:
                    raise KeyError(f"No encoder initialized for modality {modality}.")
                encoder = self.encoders[modality]
                if use_encoder_fusion_checkpoint:
                    section_latents[modality] = _checkpoint_module_forward(
                        encoder,
                        section_features[modality],
                    )
                else:
                    section_latents[modality] = encoder(section_features[modality])
                _record_forward_memory(
                    memory_recorder,
                    f"section_{section}_encoder_{modality}_end",
                    {
                        "section": section,
                        "modality": modality,
                        "latent_shape": list(section_latents[modality].shape),
                        "checkpoint_encoder_fusion": bool(checkpoint_encoder_fusion),
                    },
                )

            section_crossview_loss, section_loss_details = compute_pairwise_cosie_crossview_loss(
                section_latents,
                gamma=contrastive_gamma,
            )
            crossview_loss = crossview_loss + section_crossview_loss
            _record_forward_memory(
                memory_recorder,
                f"section_{section}_crossview_end",
                {"section": section, "modalities": list(modality_order)},
            )
            if keep_full_outputs:
                crossview_details[section] = section_loss_details
                latent_dict[section] = section_latents
            target_feature_dict[section] = section_features

            fusion_module = self.fusion_modules[combo_key]
            if use_encoder_fusion_checkpoint:
                fusion_latents = tuple(section_latents[modality] for modality in modality_order)

                def run_fusion(*latents: torch.Tensor) -> torch.Tensor:
                    return fusion_module(
                        {
                            modality: latent
                            for modality, latent in zip(modality_order, latents)
                        }
                    )

                fused = torch_checkpoint(
                    run_fusion,
                    *fusion_latents,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                fused = fusion_module(section_latents)
            _record_forward_memory(
                memory_recorder,
                f"section_{section}_fusion_end",
                {
                    "section": section,
                    "fused_shape": list(fused.shape),
                    "checkpoint_encoder_fusion": bool(checkpoint_encoder_fusion),
                },
            )
            if keep_full_outputs:
                fused_embeddings[section] = fused

            if graph_sage_cfg["enabled"]:
                graphsage_embeddings[section] = self.graphsage(fused, edge_index, edge_weight)
            else:
                graphsage_embeddings[section] = fused
            _record_forward_memory(
                memory_recorder,
                f"section_{section}_graphsage_end",
                {
                    "section": section,
                    "graphsage_enabled": bool(graph_sage_cfg["enabled"]),
                    "graphsage_shape": list(graphsage_embeddings[section].shape),
                },
            )

        if self.config["uot"]["enabled"] and self.config["ot_attention"]["enabled"]:
            if self.ot_prior is None:
                self.initialize_ot_prior(feature_dict, section_order=resolved_order)
                messages.append("Auto-initialized OT prior from preprocessed modality features.")
        elif self.ot_prior is None:
            self.ot_prior = {}

        if bidirectional_ot_attention and self.config["ot_attention"]["enabled"]:
            update_lists: dict[str, list[torch.Tensor]] = {section: [] for section in resolved_order}
            for (source_section, target_section), prior in (self.ot_prior or {}).items():
                if source_section not in update_lists or target_section not in graphsage_embeddings:
                    continue
                update = self.ot_attention.compute_update_only(
                    source_h=graphsage_embeddings[source_section],
                    target_h=graphsage_embeddings[target_section],
                    topk_idx=prior["topk_idx"],
                    topk_weight=prior["topk_weight"],
                    confidence=prior["confidence"],
                    epoch=epoch,
                    source_chunk_size=ot_attention_source_chunk_size,
                    checkpoint_attention=checkpoint_ot_attention,
                )
                _record_forward_memory(
                    memory_recorder,
                    f"ot_attention_update_{source_section}_from_{target_section}_end",
                    {
                        "source_section": source_section,
                        "target_section": target_section,
                        "source_spots": int(graphsage_embeddings[source_section].shape[0]),
                        "target_spots": int(graphsage_embeddings[target_section].shape[0]),
                        "source_chunk_size": int(ot_attention_source_chunk_size or 0),
                        "checkpoint_ot_attention": bool(checkpoint_ot_attention),
                    },
                )
                update_lists[source_section].append(update)

            for section in resolved_order:
                if update_lists[section]:
                    update_sum = update_lists[section][0]
                    for update in update_lists[section][1:]:
                        update_sum = update_sum + update
                    mean_update = update_sum / float(len(update_lists[section]))
                    final_embeddings[section] = self.ot_attention.apply_update(
                        graphsage_embeddings[section],
                        mean_update,
                    )
                    _record_forward_memory(
                        memory_recorder,
                        f"ot_attention_apply_{section}_end",
                        {"section": section, "update_count": int(len(update_lists[section]))},
                    )
                else:
                    final_embeddings[section] = graphsage_embeddings[section]
                    messages.append(f"No directional OT update found for {section}; used GraphSAGE output.")
        else:
            for source_section, target_section in zip(resolved_order[:-1], resolved_order[1:]):
                prior = (self.ot_prior or {}).get((source_section, target_section))
                if prior is None or not self.config["ot_attention"]["enabled"]:
                    final_embeddings[source_section] = graphsage_embeddings[source_section]
                    if prior is None:
                        messages.append(f"No OT prior found for {source_section}->{target_section}; used GraphSAGE output.")
                    continue
                final_embeddings[source_section] = self.ot_attention(
                    source_h=graphsage_embeddings[source_section],
                    target_h=graphsage_embeddings[target_section],
                    topk_idx=prior["topk_idx"],
                    topk_weight=prior["topk_weight"],
                    confidence=prior["confidence"],
                    epoch=epoch,
                    source_chunk_size=ot_attention_source_chunk_size,
                    checkpoint_attention=checkpoint_ot_attention,
                )
                _record_forward_memory(
                    memory_recorder,
                    f"ot_attention_update_{source_section}_from_{target_section}_end",
                    {
                        "source_section": source_section,
                        "target_section": target_section,
                        "source_spots": int(graphsage_embeddings[source_section].shape[0]),
                        "target_spots": int(graphsage_embeddings[target_section].shape[0]),
                        "source_chunk_size": int(ot_attention_source_chunk_size or 0),
                        "checkpoint_ot_attention": bool(checkpoint_ot_attention),
                    },
                )

            if resolved_order:
                last_section = resolved_order[-1]
                final_embeddings[last_section] = graphsage_embeddings[last_section]
        _record_forward_memory(
            memory_recorder,
            "ot_attention_end",
            {
                "final_embedding_shapes": {
                    section: list(embedding.shape)
                    for section, embedding in final_embeddings.items()
                },
            },
        )

        if self.config["decoder"]["enabled"] and self.config["reconstruction"]["enabled"]:
            for section in resolved_order:
                section_rec_loss, section_rec_details, section_recon = self._decode_and_reconstruct_section(
                    section=section,
                    final_embedding=final_embeddings[section],
                    target_features=target_feature_dict[section],
                    lambda_by_modality=lambda_by_modality,
                    decoder_chunk_size=decoder_chunk_size,
                    return_reconstructions=keep_full_outputs,
                    memory_recorder=memory_recorder,
                )
                reconstruction_loss = reconstruction_loss + section_rec_loss
                if keep_full_outputs:
                    reconstruction_details[section] = section_rec_details
                    reconstructions[section] = section_recon

        total_loss = lambda_reconstruction * reconstruction_loss + lambda_contrast * crossview_loss
        _record_forward_memory(
            memory_recorder,
            "end",
            {
                "decoder_enabled": bool(self.config["decoder"]["enabled"]),
                "reconstruction_enabled": bool(self.config["reconstruction"]["enabled"]),
            },
        )

        if training_loss_only:
            return {
                "losses": {
                    "total_loss": total_loss.float(),
                    "crossview_loss": crossview_loss.float(),
                    "reconstruction_loss": reconstruction_loss.float(),
                },
                "loss_scalars": {
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "crossview_loss": float(crossview_loss.detach().cpu().item()),
                    "reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
                },
                "messages": messages,
            }

        return {
            "fused_embeddings": fused_embeddings,
            "graphsage_embeddings": graphsage_embeddings,
            "final_embeddings": final_embeddings,
            "latent_dict": latent_dict,
            "reconstructions": reconstructions,
            "spatial_graph_dict": spatial_graph_dict,
            "ot_prior": self.ot_prior,
            "losses": {
                "total_loss": total_loss,
                "crossview_loss": crossview_loss,
                "reconstruction_loss": reconstruction_loss,
            },
            "loss_details": {
                "crossview": crossview_details,
                "reconstruction": reconstruction_details,
            },
            "messages": messages,
        }
