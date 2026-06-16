"""Candidate target search for scalable sparse OT priors.

FAISS is imported lazily so environments without FAISS can still import and
compile the rest of the project.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np


def _as_float32_array(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {arr.shape}.")
    return np.ascontiguousarray(arr)


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _effective_ivf_params(n_target: int, requested_nlist: int, requested_nprobe: int):
    effective_nlist = min(int(requested_nlist), max(1, int(np.sqrt(n_target)) * 4))
    effective_nlist = min(effective_nlist, n_target)
    effective_nprobe = min(int(requested_nprobe), effective_nlist)
    return effective_nlist, effective_nprobe


def _load_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for candidate_backend='faiss_ivf' or 'faiss_flat'. "
            "Install faiss-cpu or faiss-gpu in the active environment, or use "
            "candidate_backend='blockwise'."
        ) from exc
    return faiss


def _maybe_move_index_to_gpu(index, faiss, faiss_device: str):
    device_used = "cpu"
    if faiss_device not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported faiss_device: {faiss_device}")
    if faiss_device == "cpu":
        return index, device_used, None

    gpu_available = False
    try:
        gpu_available = (
            hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "index_cpu_to_gpu")
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > 0
        )
    except Exception:
        gpu_available = False

    if gpu_available:
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index), "gpu", res
        except Exception as exc:
            if faiss_device == "gpu":
                raise RuntimeError(f"Requested FAISS GPU, but moving index to GPU failed: {exc}") from exc
            return index, device_used, None

    if faiss_device == "gpu":
        raise RuntimeError("Requested FAISS GPU, but FAISS GPU API or GPU devices are unavailable.")
    return index, device_used, None


def _build_faiss_index(
    target: np.ndarray,
    backend: str,
    nlist: int,
    nprobe: int,
    faiss_device: str,
    train_sample_size: int,
    seed: int,
):
    faiss = _load_faiss()
    n_target, dim = target.shape
    metadata: dict[str, Any] = {
        "backend": backend,
        "requested_nlist": int(nlist),
        "requested_nprobe": int(nprobe),
        "embedding_dim": int(dim),
        "n_target": int(n_target),
        "faiss_device_requested": faiss_device,
    }

    if backend == "faiss_flat":
        index = faiss.IndexFlatIP(dim)
        metadata.update(
            {
                "effective_nlist": None,
                "effective_nprobe": None,
                "index_type": "IndexFlatIP",
                "requested_train_sample_size": int(train_sample_size),
                "effective_train_sample_size": 0,
                "train_sample_size": 0,
            }
        )
    elif backend == "faiss_ivf":
        effective_nlist, effective_nprobe = _effective_ivf_params(n_target, nlist, nprobe)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        requested_train = int(train_sample_size)
        if requested_train <= 0:
            raise ValueError("train_sample_size must be positive for faiss_ivf.")
        sample_size = min(requested_train, n_target)
        if sample_size < effective_nlist:
            sample_size = effective_nlist
        if sample_size < n_target:
            sample_idx = rng.choice(n_target, size=sample_size, replace=False)
            train_data = target[np.sort(sample_idx)]
        else:
            train_data = target
        index.train(np.ascontiguousarray(train_data))
        index.nprobe = effective_nprobe
        metadata.update(
            {
                "effective_nlist": int(effective_nlist),
                "effective_nprobe": int(effective_nprobe),
                "index_type": "IndexIVFFlat",
                "requested_train_sample_size": int(requested_train),
                "effective_train_sample_size": int(sample_size),
                "train_sample_size": int(sample_size),
            }
        )
    else:
        raise ValueError(f"Unsupported FAISS backend: {backend}")

    index.add(target)
    index, device_used, faiss_resource = _maybe_move_index_to_gpu(index, faiss, faiss_device)
    metadata["faiss_device_used"] = device_used
    return index, metadata, faiss_resource


def _release_faiss_resources(index=None, faiss_resource=None, device_used: str | None = None):
    released = False
    cuda_empty_cache = False
    try:
        del index
        released = True
    except Exception:
        pass
    try:
        del faiss_resource
        released = True
    except Exception:
        pass

    gc.collect()
    if device_used == "gpu":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_empty_cache = True
        except Exception:
            cuda_empty_cache = False
    return released, cuda_empty_cache


def _search_index_batched(
    index,
    source: np.ndarray,
    effective_k: int,
    query_batch_size: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | None]]:
    n_source = int(source.shape[0])
    candidate_score = np.empty((n_source, effective_k), dtype=np.float32)
    candidate_idx = np.empty((n_source, effective_k), dtype=np.int64)

    requested_batch = None if query_batch_size is None else int(query_batch_size)
    if requested_batch is None or requested_batch <= 0:
        effective_batch = n_source
    else:
        effective_batch = min(requested_batch, n_source)
    n_batches = int(np.ceil(n_source / max(effective_batch, 1)))

    for start in range(0, n_source, effective_batch):
        end = min(start + effective_batch, n_source)
        score_batch, idx_batch = index.search(np.ascontiguousarray(source[start:end]), effective_k)
        candidate_score[start:end] = score_batch.astype(np.float32, copy=False)
        candidate_idx[start:end] = idx_batch.astype(np.int64, copy=False)

    return candidate_idx, candidate_score, {
        "requested_query_batch_size": requested_batch,
        "effective_query_batch_size": int(effective_batch),
        "n_query_batches": int(n_batches),
    }


def _blockwise_exact_search(
    source: np.ndarray,
    target: np.ndarray,
    candidate_k: int,
    block_size: int = 512,
):
    import torch

    source_t = torch.as_tensor(source, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    k = min(int(candidate_k), int(target_t.shape[0]))
    all_scores = []
    all_indices = []
    for start in range(0, source_t.shape[0], block_size):
        end = min(start + block_size, source_t.shape[0])
        scores = source_t[start:end] @ target_t.t()
        values, indices = torch.topk(scores, k=k, dim=1)
        all_scores.append(values.cpu().numpy().astype(np.float32))
        all_indices.append(indices.cpu().numpy().astype(np.int64))
    return np.vstack(all_indices), np.vstack(all_scores)


def build_faiss_candidates(
    source_embeddings,
    target_embeddings,
    candidate_k: int,
    backend: str = "faiss_ivf",
    nlist: int = 4096,
    nprobe: int = 64,
    faiss_device: str = "auto",
    train_sample_size: int = 100000,
    query_batch_size: int | None = 8192,
    seed: int = 42,
) -> dict[str, Any]:
    """Find candidate targets using cosine similarity implemented as inner product."""

    start_time = time.time()
    source = _l2_normalize_np(_as_float32_array(source_embeddings))
    target = _l2_normalize_np(_as_float32_array(target_embeddings))
    n_source, dim = source.shape
    n_target = target.shape[0]
    if target.shape[1] != dim:
        raise ValueError(f"source and target dimensions differ: {dim} vs {target.shape[1]}.")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive.")
    effective_k = min(int(candidate_k), int(n_target))

    if backend == "blockwise":
        candidate_idx, candidate_score = _blockwise_exact_search(source, target, effective_k)
        metadata = {
            "backend": "blockwise",
            "requested_nlist": int(nlist),
            "effective_nlist": None,
            "requested_nprobe": int(nprobe),
            "effective_nprobe": None,
            "candidate_k": int(effective_k),
            "embedding_dim": int(dim),
            "n_source": int(n_source),
            "n_target": int(n_target),
            "faiss_device_requested": faiss_device,
            "faiss_device_used": "none",
            "index_type": "blockwise_exact_inner_product",
            "requested_train_sample_size": int(train_sample_size),
            "effective_train_sample_size": 0,
            "train_sample_size": 0,
            "requested_query_batch_size": None if query_batch_size is None else int(query_batch_size),
            "effective_query_batch_size": None,
            "n_query_batches": None,
            "faiss_resources_released": False,
            "cuda_empty_cache_after_faiss": False,
        }
    elif backend in {"faiss_ivf", "faiss_flat"}:
        index, metadata, faiss_resource = _build_faiss_index(
            target=target,
            backend=backend,
            nlist=nlist,
            nprobe=nprobe,
            faiss_device=faiss_device,
            train_sample_size=train_sample_size,
            seed=seed,
        )
        try:
            candidate_idx, candidate_score, query_metadata = _search_index_batched(
                index=index,
                source=source,
                effective_k=effective_k,
                query_batch_size=query_batch_size,
            )
        except Exception as exc:
            device_used = metadata.get("faiss_device_used")
            if device_used == "gpu":
                raise RuntimeError(
                    "FAISS GPU search failed. Try lowering --faiss_query_batch_size "
                    "or explicitly rerun with --faiss_device cpu if CPU fallback is intended. "
                    "No automatic CPU fallback is performed after GPU search failure. "
                    f"Original error: {exc}"
                ) from exc
            else:
                raise
        metadata.update(
            {
                "candidate_k": int(effective_k),
                "n_source": int(n_source),
                "n_target": int(n_target),
                **query_metadata,
            }
        )
        device_used = metadata.get("faiss_device_used")
        index_to_release = index
        faiss_resource_to_release = faiss_resource
        index = None
        faiss_resource = None
        released, cuda_empty_cache = _release_faiss_resources(
            index=index_to_release,
            faiss_resource=faiss_resource_to_release,
            device_used=device_used,
        )
        index_to_release = None
        faiss_resource_to_release = None
        gc.collect()
        if device_used == "gpu":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    cuda_empty_cache = True
            except Exception:
                pass
        metadata["faiss_resources_released"] = bool(released)
        metadata["cuda_empty_cache_after_faiss"] = bool(cuda_empty_cache)
    else:
        raise ValueError(f"Unsupported candidate backend: {backend}")

    metadata["candidate_search_time_sec"] = float(time.time() - start_time)
    return {
        "candidate_idx": candidate_idx,
        "candidate_score": candidate_score,
        "metadata": metadata,
    }
