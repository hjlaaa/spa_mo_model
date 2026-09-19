"""Existing positional, RNG and section/barcode sampling contracts."""
from __future__ import annotations
from typing import Any
import hashlib
import numpy as np
import pandas as pd


def sample_indices(n: int, size: int, seed: int) -> np.ndarray:
    if size <= 0 or size >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def permutation_sample_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.permutation(n)[: min(sample_size, n)])


def section_barcode_sample(methods: list[Any], *, section_order, section_counts, sample_counts, sample_count, seed) -> pd.DataFrame:
    reference = pd.DataFrame(
        {
            "section": methods[0].sections,
            "obs_name": methods[0].barcodes,
        }
    )
    section_rank = {section: rank for rank, section in enumerate(section_order)}
    reference["section_rank"] = reference["section"].map(section_rank)
    reference = reference.sort_values(
        ["section_rank", "obs_name"], kind="mergesort"
    ).drop(columns="section_rank")
    reference_index = pd.MultiIndex.from_frame(reference)
    for method in methods[1:]:
        candidate = pd.MultiIndex.from_arrays(
            [method.sections, method.barcodes],
            names=["section", "obs_name"],
        )
        if len(candidate) != len(reference_index) or set(candidate) != set(
            reference_index
        ):
            raise ValueError(
                f"{method.name}: section/barcode universe differs from other methods"
            )

    rng = np.random.default_rng(seed)
    rows = []
    sample_order = 0
    for section in section_order:
        candidates = reference[reference["section"] == section].reset_index(
            drop=True
        )
        chosen = np.sort(
            rng.choice(
                len(candidates),
                size=sample_counts[section],
                replace=False,
            )
        )
        selected = candidates.iloc[chosen]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "sample_order": sample_order,
                    "section": row["section"],
                    "obs_name": row["obs_name"],
                    "section_n_total": section_counts[section],
                    "section_n_sampled": sample_counts[section],
                    "sampling_seed": seed,
                }
            )
            sample_order += 1
    result = pd.DataFrame(rows)
    if len(result) != sample_count or result[
        ["section", "obs_name"]
    ].duplicated().any():
        raise ValueError("invalid fixed ASW sample")
    return result


def section_sample_mask(data: Any, sample: pd.DataFrame, *, sample_counts, sample_count) -> np.ndarray:
    selected = set(zip(sample["section"], sample["obs_name"]))
    mask = np.fromiter(
        (
            (section, barcode) in selected
            for section, barcode in zip(data.sections, data.barcodes)
        ),
        dtype=bool,
        count=len(data.embedding),
    )
    if int(mask.sum()) != sample_count:
        raise ValueError(f"{data.name}: fixed ASW sample does not map to 10000 spots")
    for section, expected in sample_counts.items():
        if int(np.sum(mask & (data.sections == section))) != expected:
            raise ValueError(f"{data.name}: ASW stratum mismatch for {section}")
    return mask


def barcode_hash(prefix: str, barcode: str) -> str:
    return hashlib.sha256(f"{prefix}{barcode}".encode("utf-8")).hexdigest()


def select_by_barcode_hash(
    reference: Any,
    counts: dict[str, int],
    prefix: str,
    *, section_order, section_counts,
) -> pd.DataFrame:
    rows = []
    for section in section_order:
        barcodes = reference.barcodes[reference.sections == section]
        frame = pd.DataFrame({"section": section, "obs_name": barcodes})
        frame["barcode_sha256"] = [
            barcode_hash(prefix, value) for value in frame["obs_name"]
        ]
        frame = frame.sort_values(
            ["barcode_sha256", "obs_name"], kind="mergesort"
        ).head(counts[section]).reset_index(drop=True)
        frame["sample_order_within_section"] = np.arange(len(frame))
        frame["section_n_total"] = section_counts[section]
        frame["section_n_sampled"] = counts[section]
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if (
        len(result) != sum(counts.values())
        or result[["section", "obs_name"]].duplicated().any()
    ):
        raise ValueError("invalid deterministic barcode-hash sample")
    return result


def key_mask(data: Any, sample: pd.DataFrame) -> np.ndarray:
    selected = set(zip(sample["section"], sample["obs_name"]))
    return np.fromiter(
        (
            (section, barcode) in selected
            for section, barcode in zip(data.sections, data.barcodes)
        ),
        dtype=bool,
        count=len(data.embedding),
    )


def proportional_selections(
    sections: list[str],
    counts: dict[str, int],
    maximum: int,
    seed: int,
) -> dict[str, np.ndarray]:
    total = sum(counts.values())
    if maximum <= 0 or total <= maximum:
        return {
            section: np.arange(counts[section], dtype=np.int64)
            for section in sections
        }
    exact = np.array([maximum * counts[s] / total for s in sections])
    quotas = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - quotas))[: maximum - int(quotas.sum())]:
        quotas[index] += 1
    rng = np.random.default_rng(seed)
    return {
        section: np.sort(
            rng.choice(counts[section], size=int(quota), replace=False)
        ).astype(np.int64)
        for section, quota in zip(sections, quotas)
    }
