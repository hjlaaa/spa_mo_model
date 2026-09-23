"""Shared Stage/prior/dry/fit/export runtime with explicit historical policies."""
from __future__ import annotations
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch
from model.stage_model import StageMultiModalModel
from .fit import initialize_model_ot_prior, train_small_crc_model, run_one_forward, make_forward_memory_recorder
from .artifacts import save_final_embeddings, save_spatial_arrays, save_ot_prior_topk


@dataclass(frozen=True)
class ExecutionPolicy:
    secondary_modality: str
    prior_checks: Literal["paired", "adjacent"]
    dry_eval: bool
    detailed_dry_memory: bool
    paired_output_checks: bool


def execute_prepared_task(run_config, prepared, *, memory_monitor, policy: ExecutionPolicy):
    """No input reading/resolving or epoch loop; callers retain summary layout."""
    args = Namespace()
    args.__dict__ = run_config["runner"]
    feature_dict = prepared.feature_dict
    spatial_loc_dict = prepared.spatial_loc_dict
    processed_data_dict = prepared.processed_data_dict
    section_order = prepared.section_order
    output_dir = Path(args.output_dir)
    secondary_modality = policy.secondary_modality
    model_config = run_config["model_config"]
    model, _, resolved_modality_order = initialize_prepared_task(
        model_config, prepared, args, memory_monitor=memory_monitor,
        expected_modalities=["RNA", secondary_modality],
    )
    del _  # These callers originally discarded the full initializer return.
    if policy.prior_checks == "paired":
        first_section, second_section = run_config["section_order"]
        initial_prior = model.ot_prior[(first_section, second_section)]
        expected_keys = {(first_section, second_section), (second_section, first_section)}
        actual_keys = set((model.ot_prior or {}).keys())
        if not expected_keys.issubset(actual_keys):
            raise ValueError(f"Bidirectional OT prior is missing direction keys: {expected_keys - actual_keys}")
        initial_ot_modalities_used = list(initial_prior.get("modalities_used", []))
        if initial_prior.get("metadata", {}).get("ot_prior_mode") != "candidate_sparse":
            raise ValueError("Expected candidate_sparse initial OT prior metadata.")
        first_pair = (first_section, second_section)
    else:
        first_pair = (section_order[0], section_order[1]) if len(section_order) > 1 else None
        if first_pair is None:
            raise ValueError("MISAR run requires at least two sections.")
        initial_prior = model.ot_prior[first_pair]
        initial_ot_modalities_used = list(initial_prior.get("modalities_used", []))
        expected_keys = {
            key
            for left, right in zip(section_order[:-1], section_order[1:])
            for key in [(left, right), (right, left)]
        }
        actual_keys = set((model.ot_prior or {}).keys())
        if not expected_keys.issubset(actual_keys):
            raise ValueError(f"Bidirectional OT prior is missing keys: {expected_keys - actual_keys}")

    history = None
    ot_updates = []
    if args.train:
        history, outputs, ot_updates = train_small_crc_model(
            model, feature_dict, spatial_loc_dict, processed_data_dict,
            section_order, args, memory_monitor=memory_monitor,
        )
    else:
        if policy.dry_eval:
            model.eval()
        with torch.no_grad():
            memory_monitor.reset_peak()
            memory_monitor.record("dry_run_forward_start", epoch=0)
            forward_options = dict(
                epoch=0,
                decoder_chunk_size=int(args.decoder_chunk_size),
                ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
                cache_spatial_graphs=bool(args.cache_spatial_graphs),
                checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
                checkpoint_encoder_fusion=bool(args.checkpoint_encoder_fusion),
                checkpoint_decoder_chunks=bool(args.checkpoint_decoder_chunks),
                checkpoint_graph_encoder=bool(args.checkpoint_graph_encoder),
            )
            if policy.detailed_dry_memory:
                forward_options["memory_recorder"] = make_forward_memory_recorder(
                    memory_monitor, bool(args.log_cuda_memory_detail), 0, "dry_run_forward",
                )
            outputs = run_one_forward(
                model, feature_dict, spatial_loc_dict, processed_data_dict,
                section_order, **forward_options,
            )
            memory_monitor.record("dry_run_forward_end", epoch=0)
    if policy.paired_output_checks:
        prior = outputs["ot_prior"][(first_section, second_section)]
        reconstruction_keys = {
            section: sorted(modalities.keys())
            for section, modalities in outputs["reconstructions"].items()
        }
        ot_modalities_used = list(prior.get("modalities_used", []))
        total_loss_finite = bool(torch.isfinite(outputs["losses"]["total_loss"]).item())
        if reconstruction_keys != {first_section: ["Protein", "RNA"], second_section: ["Protein", "RNA"]}:
            raise ValueError(f"Unexpected reconstruction keys: {reconstruction_keys}")
        # Initial UOT remains multimodal RNA/Protein. Dynamic refresh preserves
        # the original detached final-embedding source by default.
        accepted_ot_modalities = [["RNA", "Protein"]]
        if args.train and ot_updates:
            accepted_ot_modalities.append(["final_embedding"])
            accepted_ot_modalities.append(["ot_embedding"])
        accepted_ot_modalities.append(["RNA", "Protein"])
        accepted_ot_modalities.append(["ot_embedding"])
        if ot_modalities_used not in accepted_ot_modalities:
            raise ValueError(f"Unexpected OT modalities_used: {ot_modalities_used}")
        reverse_prior = outputs["ot_prior"].get((second_section, first_section))
        if reverse_prior is None:
            raise ValueError(f"Bidirectional OT prior is missing {second_section}<-{first_section} direction.")
        if int(prior["topk_idx"].max().item()) >= int(feature_dict[second_section]["RNA"].shape[0]):
            raise ValueError(f"{first_section}<-{second_section} topk_idx contains out-of-range target indices.")
        if int(reverse_prior["topk_idx"].max().item()) >= int(feature_dict[first_section]["RNA"].shape[0]):
            raise ValueError(f"{second_section}<-{first_section} topk_idx contains out-of-range target indices.")
        if not total_loss_finite:
            raise ValueError("total_loss is not finite.")

    else:
        reconstruction_keys = {
            section: sorted(modalities.keys())
            for section, modalities in outputs["reconstructions"].items()
        }
        expected_reconstruction_keys = {section: [secondary_modality, "RNA"] for section in section_order}
        if reconstruction_keys != expected_reconstruction_keys:
            raise ValueError(f"Unexpected reconstruction keys: {reconstruction_keys}")
        total_loss_finite = bool(torch.isfinite(outputs["losses"]["total_loss"]).item())
        if not total_loss_finite:
            raise ValueError("total_loss is not finite.")

        ot_modalities_used = None
    embedding_paths = {}
    spatial_paths = {}
    ot_prior_topk_files = {}
    if args.save_embeddings:
        memory_monitor.reset_peak()
        memory_monitor.record("save_embeddings_start")
        embedding_paths = save_final_embeddings(output_dir, outputs["final_embeddings"])
        memory_monitor.record("save_embeddings_end")
    if args.save_outputs:
        memory_monitor.reset_peak()
        memory_monitor.record("save_spatial_arrays_start")
        spatial_paths = save_spatial_arrays(output_dir, spatial_loc_dict)
        memory_monitor.record("save_spatial_arrays_end")
    if args.save_ot_prior_topk:
        memory_monitor.reset_peak()
        memory_monitor.record("save_ot_prior_topk_start")
        ot_prior_topk_files = save_ot_prior_topk(
            output_dir / "ot_prior_topk",
            outputs.get("ot_prior"),
            outputs["final_embeddings"],
            "training_final_eval" if args.train else "dry_run",
            save_candidate_qc=bool(args.save_candidate_qc),
        )
        memory_monitor.record("save_ot_prior_topk_end")

    return {
        "model": model,
        "initial_prior": initial_prior,
        "outputs": outputs,
        "history": history,
        "ot_updates": ot_updates,
        "resolved_modality_order": resolved_modality_order,
        "initial_ot_modalities_used": initial_ot_modalities_used,
        "reconstruction_keys": reconstruction_keys,
        "total_loss_finite": total_loss_finite,
        "ot_modalities_used": ot_modalities_used,
        "first_pair": first_pair,
        "embedding_paths": embedding_paths,
        "spatial_paths": spatial_paths,
        "ot_prior_topk_files": ot_prior_topk_files,
    }


def initialize_prepared_task(model_config, prepared, args, *, memory_monitor=None, expected_modalities=None, defer_prior=False):
    """Initialize the same Stage/prior; only existing callers request modality checks."""
    feature_dict = prepared.feature_dict
    section_order = prepared.section_order
    if memory_monitor is not None:
        memory_monitor.reset_peak()
        memory_monitor.record("model_init_start")
    model = StageMultiModalModel(config=model_config, feature_dict=feature_dict)
    if memory_monitor is not None:
        memory_monitor.record("model_init_end")
    resolved_modality_order = None
    if expected_modalities is not None:
        resolved_modality_order = list(model._resolve_modality_order(feature_dict[section_order[0]]))
        if resolved_modality_order != expected_modalities:
            raise ValueError(f"Expected {expected_modalities}, got {resolved_modality_order}.")
    if defer_prior:
        # Some callers create their monitor/check mode before optional UOT initialization.
        return model, None, resolved_modality_order
    if memory_monitor is not None:
        memory_monitor.reset_peak()
        memory_monitor.record("initial_ot_prior_start")
    initial_prior = initialize_model_ot_prior(model, feature_dict, section_order, args)
    if memory_monitor is not None:
        memory_monitor.record("initial_ot_prior_end")
    return model, initial_prior, resolved_modality_order


@dataclass(frozen=True)
class Epoch0Policy:
    eval_mode: bool


def run_epoch0_preflight(model, prepared, *, policy: Epoch0Policy):
    """Explicit epoch0 step, also run before training; distinct from a dry branch."""
    if policy.eval_mode:
        model.eval()
    with torch.no_grad():
        return model(
            feature_dict=prepared.feature_dict,
            spatial_loc_dict=prepared.spatial_loc_dict,
            processed_data_dict=prepared.processed_data_dict,
            section_order=prepared.section_order,
            epoch=0,
        )
