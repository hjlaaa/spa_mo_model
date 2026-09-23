"""Requested input/truth projection and compatible entry into public evaluation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from analysis.protocols import AnalysisSpec
from analysis.cache import check_output_path
from analysis.evaluation_workflow import evaluate_standardized_scopes, save_labels

def truth_arrays(data: Any, key: str) -> dict[str, np.ndarray]:
    if key in {"mousebrain", "misar_seq"}:
        return {name: np.asarray(values) for name, values in data.truth.items()}
    if key == "simulation":
        return {"spatial_domain": np.asarray(data.truth)}
    return {}

def analyze_prepared(data: Any, spec: AnalysisSpec, model_version: str,
                     post_ot_graphsage_scale: float, *, input_dir: Path,
                     output_dir: Path, persisted_asw_mask=None,
                     joint_per_section: bool = False) -> dict[str, Any]:
    output = check_output_path(output_dir, [input_dir])
    raw = np.asarray(data.embedding, dtype=np.float64)
    sections = np.asarray(data.sections).astype(str)
    truth = truth_arrays(data, spec.key)
    return evaluate_standardized_scopes(
        data, spec, model_version, post_ot_graphsage_scale, output=output,
        raw=raw, sections=sections, truth=truth, input_dir=input_dir,
        persisted_asw_mask=persisted_asw_mask, joint_per_section=joint_per_section)
