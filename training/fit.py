"""Shared full-batch training runtime for the retained dataset entry points.

Callers supply prepared tensors, a constructed model/prior, and resolved
training arguments. Dataset preparation, configuration, and exports stay with
those callers.
"""
from __future__ import annotations

import gc
import json
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from model.stage_model import StageMultiModalModel, should_update_ot
from .graph_refresh import prepare_refresh_outputs


def json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu())
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024**3)


def release_python_and_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def amp_enabled(args) -> bool:
    return bool(args.device == "cuda" and args.amp_dtype != "none")


def autocast_context(args):
    if not amp_enabled(args):
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(args):
    enabled = bool(args.device == "cuda" and args.amp_dtype == "fp16")
    return torch.cuda.amp.GradScaler(enabled=enabled)


class CudaMemoryMonitor:
    """Append lightweight CUDA memory events to a JSONL file."""

    def __init__(self, enabled: bool, output_dir: Path, requested_device: str):
        self.enabled = bool(enabled)
        self.requested_device = requested_device
        self.output_path = output_dir / "cuda_memory_trace.jsonl"
        self.event_count = 0
        if self.enabled:
            self.output_path.write_text("", encoding="utf-8")

    def reset_peak(self) -> None:
        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record(
        self,
        stage: str,
        epoch: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}

        event: dict[str, Any] = {
            "event_index": int(self.event_count),
            "time_sec": float(time.time()),
            "stage": str(stage),
            "epoch": int(epoch) if epoch is not None else None,
            "requested_device": self.requested_device,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            device_index = int(torch.cuda.current_device())
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            event.update(
                {
                    "device_index": device_index,
                    "device_name": torch.cuda.get_device_name(device_index),
                    "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated(device_index)),
                    "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved(device_index)),
                    "max_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated(device_index)),
                    "max_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved(device_index)),
                    "free_gib": bytes_to_gib(free_bytes),
                    "total_gib": bytes_to_gib(total_bytes),
                }
            )
        if extra:
            event.update(json_safe(dict(extra)))

        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(event), ensure_ascii=False) + "\n")
        self.event_count += 1
        return event


def make_forward_memory_recorder(
    memory_monitor: CudaMemoryMonitor | None,
    enabled: bool,
    epoch: int,
    phase: str,
):
    if memory_monitor is None or not enabled or not memory_monitor.enabled:
        return None

    def record_forward_detail(stage: str, extra: Mapping[str, Any] | None = None) -> None:
        memory_monitor.record(f"{phase}_detail_{stage}", epoch=epoch, extra=extra)

    return record_forward_detail


def run_one_forward(
    model: StageMultiModalModel,
    feature_dict: Mapping[str, Mapping[str, torch.Tensor]],
    spatial_loc_dict: Mapping[str, Any],
    processed_data_dict: Any,
    section_order: list[str],
    epoch: int,
    training_loss_only: bool = False,
    decoder_chunk_size: int = 0,
    ot_attention_source_chunk_size: int = 0,
    cache_spatial_graphs: bool = False,
    checkpoint_ot_attention: bool = False,
    checkpoint_encoder_fusion: bool = False,
    checkpoint_decoder_chunks: bool = False,
    checkpoint_graph_encoder: bool = False,
    memory_recorder=None,
) -> dict[str, Any]:
    return model(
        feature_dict=feature_dict,
        spatial_loc_dict=spatial_loc_dict,
        processed_data_dict=processed_data_dict,
        section_order=section_order,
        epoch=epoch,
        training_loss_only=training_loss_only,
        decoder_chunk_size=decoder_chunk_size,
        ot_attention_source_chunk_size=ot_attention_source_chunk_size,
        cache_spatial_graphs=cache_spatial_graphs,
        checkpoint_ot_attention=checkpoint_ot_attention,
        checkpoint_encoder_fusion=checkpoint_encoder_fusion,
        checkpoint_decoder_chunks=checkpoint_decoder_chunks,
        checkpoint_graph_encoder=checkpoint_graph_encoder,
        memory_recorder=memory_recorder,
    )


def sparse_prior_kwargs(args) -> dict[str, Any]:
    return {
        "candidate_k": int(args.candidate_k),
        "attention_topk": int(args.attention_topk),
        "candidate_backend": args.candidate_backend,
        "faiss_nlist": int(args.faiss_nlist),
        "faiss_nprobe": int(args.faiss_nprobe),
        "faiss_device": args.faiss_device,
        "faiss_train_sample_size": int(args.faiss_train_sample_size),
        "faiss_query_batch_size": (
            int(args.faiss_query_batch_size)
            if args.faiss_query_batch_size is not None
            else None
        ),
        "seed": int(args.seed),
        "epsilon": float(args.uot_epsilon),
        "tau_a": float(args.uot_tau_a),
        "tau_b": float(args.uot_tau_b),
        "max_iter": int(args.uot_max_iter),
        "stabilizer": float(args.uot_stabilizer),
    }


def initialize_model_ot_prior(
    model: StageMultiModalModel,
    feature_dict: Mapping[str, Mapping[str, torch.Tensor]],
    section_order: list[str],
    args,
):
    return model.initialize_candidate_sparse_ot_prior(
        feature_dict,
        section_order=section_order,
        initial_modality_candidate_k=int(args.initial_modality_candidate_k),
        **sparse_prior_kwargs(args),
    )


def update_model_ot_prior(
    model: StageMultiModalModel,
    eval_outputs: Mapping[str, Any],
    section_order: list[str],
    args,
):
    embeddings, context_embeddings, _ = model.prepare_ot_prior_refresh(
        eval_outputs,
    )
    uot_cfg = model.config["uot"]
    topology_weight = (
        float(uot_cfg.get("topology_context_weight", 0.0))
        if bool(uot_cfg.get("topology_aware_refresh_enabled", False))
        else 0.0
    )
    return model.update_candidate_sparse_ot_prior(
        embeddings,
        section_order=section_order,
        context_embedding_dict=context_embeddings,
        topology_context_weight=topology_weight,
        **sparse_prior_kwargs(args),
    )


def lambda_for_epoch(epoch: int, schedule, default_lambda: float) -> float:
    if schedule is None:
        return float(default_lambda)
    for start_epoch, end_epoch, value in schedule:
        if start_epoch <= epoch <= end_epoch:
            return float(value)
    return float(default_lambda)


def train_small_crc_model(
    model: StageMultiModalModel,
    feature_dict: Mapping[str, Mapping[str, torch.Tensor]],
    spatial_loc_dict: Mapping[str, Any],
    processed_data_dict: Any,
    section_order: list[str],
    args,
    memory_monitor: CudaMemoryMonitor | None = None,
    *,
    lambda_contrast_schedule=None,
    clear_step_state: bool = True,
    record_elapsed_time: bool = True,
    allow_empty_epochs: bool = False,
) -> tuple[list[dict[str, float]], dict[str, Any], list[int]]:
    # Existing callers consume the final result from the same epoch runtime.
    for _, history, final_outputs, ot_updates in iter_fit_model(
        model, feature_dict, spatial_loc_dict, processed_data_dict, section_order,
        args, memory_monitor,
        lambda_contrast_schedule=lambda_contrast_schedule,
        clear_step_state=clear_step_state,
        record_elapsed_time=record_elapsed_time,
        allow_empty_epochs=allow_empty_epochs,
    ):
        pass
    return history, final_outputs, ot_updates


def iter_fit_model(
    model: StageMultiModalModel,
    feature_dict: Mapping[str, Mapping[str, torch.Tensor]],
    spatial_loc_dict: Mapping[str, Any],
    processed_data_dict: Any,
    section_order: list[str],
    args,
    memory_monitor: CudaMemoryMonitor | None = None,
    *,
    lambda_contrast_schedule=None,
    clear_step_state: bool = True,
    record_elapsed_time: bool = True,
    allow_empty_epochs: bool = False,
    record_loss_weights: bool = True,
    refresh_ot: bool = True,
    yield_every: int = 0,
) -> Iterator[tuple[bool, list[dict[str, float]], dict[str, Any], list[int]]]:
    """Yield (is_final, history, outputs, ot_updates) for caller-owned saves.

    Intermediate results follow completed steps and any scheduled refresh.
    Their outputs are from that epoch's training forward. The final result
    follows final eval. No autocast/no_grad context remains active at a yield.
    """
    # Defaults preserve the CRC runtime. Callers can retain step gradients and
    # outputs, omit timing metadata, or request a resolved lambda schedule.
    epochs = int(args.epochs)
    if epochs <= 0 and not allow_empty_epochs:
        raise ValueError("--epochs must be positive when --train is set.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = make_grad_scaler(args)
    history: list[dict[str, float]] = []
    ot_updates: list[int] = []
    feature_graph_refreshes: list[dict[str, Any]] = []
    default_lambda = float(model.config["loss"]["lambda_contrast"])
    for epoch in range(1, epochs + 1):
        if record_elapsed_time:
            start_time = time.time()
        model.train()
        if lambda_contrast_schedule is not None:
            model.config["loss"]["lambda_contrast"] = lambda_for_epoch(
                epoch, lambda_contrast_schedule, default_lambda,
            )
        if memory_monitor is not None:
            memory_monitor.record("epoch_start", epoch=epoch)
            memory_monitor.reset_peak()
            memory_monitor.record("epoch_forward_start", epoch=epoch)
        with autocast_context(args):
            outputs = run_one_forward(
                model,
                feature_dict,
                spatial_loc_dict,
                processed_data_dict,
                section_order,
                epoch=epoch,
                training_loss_only=bool(args.training_loss_only),
                decoder_chunk_size=int(args.decoder_chunk_size),
                ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
                cache_spatial_graphs=bool(args.cache_spatial_graphs),
                checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
                checkpoint_encoder_fusion=bool(args.checkpoint_encoder_fusion),
                checkpoint_decoder_chunks=bool(args.checkpoint_decoder_chunks),
                checkpoint_graph_encoder=bool(args.checkpoint_graph_encoder),
                memory_recorder=make_forward_memory_recorder(
                    memory_monitor,
                    bool(args.log_cuda_memory_detail),
                    epoch,
                    "train_forward",
                ),
            )
        forward_memory = (
            memory_monitor.record("epoch_forward_end", epoch=epoch)
            if memory_monitor is not None
            else {}
        )
        loss = outputs["losses"]["total_loss"].float()
        optimizer.zero_grad(set_to_none=True)
        if memory_monitor is not None:
            memory_monitor.reset_peak()
            memory_monitor.record("epoch_backward_start", epoch=epoch)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        backward_memory = (
            memory_monitor.record("epoch_backward_end", epoch=epoch)
            if memory_monitor is not None
            else {}
        )
        if memory_monitor is not None:
            memory_monitor.reset_peak()
            memory_monitor.record("epoch_optimizer_step_start", epoch=epoch)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if clear_step_state:
            optimizer.zero_grad(set_to_none=True)
        optimizer_memory = (
            memory_monitor.record("epoch_optimizer_step_end", epoch=epoch)
            if memory_monitor is not None
            else {}
        )

        total_loss_value = float(loss.detach().cpu().item())
        crossview_loss_value = float(outputs["losses"]["crossview_loss"].detach().cpu().item())
        reconstruction_loss_value = float(outputs["losses"]["reconstruction_loss"].detach().cpu().item())
        record = {"epoch": int(epoch)}
        if record_loss_weights:
            record["lambda_contrast"] = float(model.config["loss"]["lambda_contrast"])
        record.update(
            total_loss=total_loss_value,
            crossview_loss=crossview_loss_value,
            reconstruction_loss=reconstruction_loss_value,
        )
        if record_elapsed_time:
            record["elapsed_time_sec"] = float(time.time() - start_time)
        if record_loss_weights:
            record["weighted_crossview_loss"] = record["lambda_contrast"] * record["crossview_loss"]
        if memory_monitor is not None:
            stage_events = [forward_memory, backward_memory, optimizer_memory]
            record.update(
                {
                    "cuda_allocated_gib": optimizer_memory.get("allocated_gib"),
                    "cuda_reserved_gib": optimizer_memory.get("reserved_gib"),
                    "cuda_epoch_max_allocated_gib": max(
                        (
                            event.get("max_allocated_gib", 0.0) or 0.0
                            for event in stage_events
                        ),
                        default=0.0,
                    ),
                    "cuda_epoch_max_reserved_gib": max(
                        (
                            event.get("max_reserved_gib", 0.0) or 0.0
                            for event in stage_events
                        ),
                        default=0.0,
                    ),
                    "cuda_forward_max_allocated_gib": forward_memory.get("max_allocated_gib"),
                    "cuda_backward_max_allocated_gib": backward_memory.get("max_allocated_gib"),
                    "cuda_optimizer_max_allocated_gib": optimizer_memory.get("max_allocated_gib"),
                }
            )
        history.append(record)

        if args.log_every > 0 and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == epochs):
            if record_loss_weights:
                print(
                    f"epoch={epoch} total={record['total_loss']:.6f} "
                    f"lambda_contrast={record['lambda_contrast']:.6g} "
                    f"weighted_crossview={record['weighted_crossview_loss']:.6f} "
                    f"crossview={record['crossview_loss']:.6f} "
                    f"reconstruction={record['reconstruction_loss']:.6f}"
                )
            else:
                print(
                    f"epoch={epoch} total={record['total_loss']:.6f} "
                    f"crossview={record['crossview_loss']:.6f} "
                    f"reconstruction={record['reconstruction_loss']:.6f}"
                )

        if yield_every > 0:
            epoch_outputs = outputs
        if memory_monitor is not None:
            memory_monitor.record("epoch_cleanup_start", epoch=epoch)
        if clear_step_state:
            del outputs
            del loss
            release_python_and_cuda_cache()
        if memory_monitor is not None:
            memory_monitor.record("epoch_cleanup_end", epoch=epoch)

        if refresh_ot and should_update_ot(epoch, int(args.update_interval)):
            model.eval()
            with torch.no_grad():
                if memory_monitor is not None:
                    memory_monitor.reset_peak()
                    memory_monitor.record("ot_update_forward_start", epoch=epoch)
                refresh_forward = partial(
                    run_one_forward,
                    model,
                    feature_dict,
                    spatial_loc_dict,
                    processed_data_dict,
                    section_order,
                    epoch=epoch,
                    decoder_chunk_size=int(args.decoder_chunk_size),
                    ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
                    cache_spatial_graphs=bool(args.cache_spatial_graphs),
                    checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
                    checkpoint_encoder_fusion=bool(args.checkpoint_encoder_fusion),
                    checkpoint_decoder_chunks=bool(args.checkpoint_decoder_chunks),
                    checkpoint_graph_encoder=bool(args.checkpoint_graph_encoder),
                    memory_recorder=make_forward_memory_recorder(
                        memory_monitor,
                        bool(args.log_cuda_memory_detail),
                        epoch,
                        "ot_update_forward",
                    ),
                )
                eval_outputs, feature_refresh = prepare_refresh_outputs(
                    model, refresh_forward, epoch=epoch,
                )
                if memory_monitor is not None:
                    memory_monitor.record("ot_update_forward_end", epoch=epoch)
                    memory_monitor.reset_peak()
                    memory_monitor.record("ot_update_prior_start", epoch=epoch)
                update_model_ot_prior(model, eval_outputs, section_order, args)
                if feature_refresh is not None:
                    feature_graph_refreshes.append(feature_refresh)
                    if getattr(args, "output_dir", None):
                        trace = Path(args.output_dir) / "feature_graph_refresh.json"
                        trace.parent.mkdir(parents=True, exist_ok=True)
                        trace.write_text(json.dumps(feature_graph_refreshes, indent=2), encoding="utf-8")
                if memory_monitor is not None:
                    memory_monitor.record("ot_update_prior_end", epoch=epoch)
                if clear_step_state:
                    del eval_outputs
                    release_python_and_cuda_cache()
                if memory_monitor is not None:
                    memory_monitor.record("ot_update_cleanup_end", epoch=epoch)
            ot_updates.append(epoch)
            print(f"Updated OT prior at epoch {epoch}.")

        if yield_every > 0 and epoch % yield_every == 0:
            yield False, history, epoch_outputs, ot_updates

    model.eval()
    if lambda_contrast_schedule is not None:
        model.config["loss"]["lambda_contrast"] = lambda_for_epoch(
            epochs, lambda_contrast_schedule, default_lambda,
        )
    with torch.no_grad():
        if memory_monitor is not None:
            memory_monitor.reset_peak()
            memory_monitor.record("final_eval_start", epoch=epochs)
        final_outputs = run_one_forward(
            model,
            feature_dict,
            spatial_loc_dict,
            processed_data_dict,
            section_order,
            epoch=epochs,
            decoder_chunk_size=int(args.decoder_chunk_size),
            ot_attention_source_chunk_size=int(args.ot_attention_source_chunk_size),
            cache_spatial_graphs=bool(args.cache_spatial_graphs),
            checkpoint_ot_attention=bool(args.checkpoint_ot_attention),
            checkpoint_encoder_fusion=bool(args.checkpoint_encoder_fusion),
            checkpoint_decoder_chunks=bool(args.checkpoint_decoder_chunks),
            checkpoint_graph_encoder=bool(args.checkpoint_graph_encoder),
            memory_recorder=make_forward_memory_recorder(
                memory_monitor,
                bool(args.log_cuda_memory_detail),
                epochs,
                "final_eval_forward",
            ),
        )
        if memory_monitor is not None:
            memory_monitor.record("final_eval_end", epoch=epochs)
    if model.feature_graph.enabled:
        final_outputs["feature_graph_refreshes"] = feature_graph_refreshes
    yield True, history, final_outputs, ot_updates
