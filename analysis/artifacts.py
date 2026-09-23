"""Validate completed requested analyses using each workflow's output contract."""
from __future__ import annotations

import json
from pathlib import Path

from .protocols import SPECS


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty analysis artifact: {path}")


def requested_analysis_artifacts(
    dataset: str,
    output_dir: Path,
    *,
    model_version: str | None = None,
    post_ot_graphsage_scale: float | None = None,
    k_values: list[int] | None = None,
    require_plots: bool = False,
) -> dict[str, dict[str, int]]:
    """Return file identities after validating a completed requested analysis.

    The generic requested workflow writes into ``standardized_embedding``;
    SPATCH writes into the requested output directory itself. Caller supplied
    expectations are checked without tying the contract to an experiment name.
    """
    output = Path(output_dir).resolve()
    if dataset == "spatch":
        base = output
        completion_path = base / "analysis_completion_manifest.json"
        display_name = "SPATCH"
    elif dataset in SPECS:
        base = output / "standardized_embedding"
        completion_path = base / "analysis_completion.json"
        display_name = SPECS[dataset].display_name
    else:
        raise ValueError(f"No requested analysis artifact contract for {dataset}")

    config_path = base / "config.json"
    source_path = base / "analysis_source_manifest.json"
    required = [config_path, completion_path, source_path]
    for path in required:
        _nonempty(path)
    config = _json(config_path)
    completion = _json(completion_path)
    source = _json(source_path)
    if source.get("complete") is not True:
        raise ValueError(f"Analysis source manifest is not complete: {source_path}")
    if config.get("dataset") != display_name or completion.get("dataset") != display_name:
        raise ValueError(f"Unexpected analysis dataset: {base}")
    if Path(completion.get("analysis", "")).resolve() != base:
        raise ValueError(f"Analysis completion path does not match output: {base}")
    if model_version is not None and config.get("model_version") != model_version:
        raise ValueError(f"Unexpected analysis model version: {base}")
    if post_ot_graphsage_scale is not None and (
        config.get("post_ot_graphsage_scale") != post_ot_graphsage_scale
    ):
        raise ValueError(f"Unexpected post-OT GraphSAGE scale: {base}")

    if dataset == "spatch":
        if completion.get("status") != "complete":
            raise ValueError(f"SPATCH analysis is not complete: {base}")
        if model_version is not None and completion.get("model_version") != model_version:
            raise ValueError(f"Unexpected SPATCH completion version: {base}")
        if post_ot_graphsage_scale is not None and (
            completion.get("post_ot_graphsage_scale") != post_ot_graphsage_scale
        ):
            raise ValueError(f"Unexpected SPATCH completion scale: {base}")
        if k_values is not None and completion.get("k_values") != list(k_values):
            raise ValueError(f"Unexpected SPATCH K values: {base}")
    elif k_values is not None:
        actual = [completion.get(key) for key in ("joint_k_values", "independent_k_values")
                  if completion.get(key)]
        if not actual or any(values != list(k_values) for values in actual):
            raise ValueError(f"Unexpected requested K values: {base}")

    for name in ("internal_metrics", "batch_metrics", "supervised_metrics"):
        value = completion.get(name)
        if value is None:
            if name != "supervised_metrics" or dataset == "spatch":
                raise ValueError(f"Missing {name} in analysis completion: {base}")
            continue
        metric = base / "metrics" / f"{name}.csv"
        if Path(value).resolve() != metric:
            raise ValueError(f"Unexpected {name} path: {value}")
        _nonempty(metric)
        required.append(metric)

    if require_plots:
        figures = base if dataset == "spatch" else output / "figures"
        plots = list(figures.rglob("spatial_*.png"))
        if not plots:
            raise ValueError(f"Missing requested spatial plots: {figures}")
        if dataset == "spatch":
            if completion.get("spatial_plot_count") != len(plots):
                raise ValueError(f"SPATCH spatial plot count mismatch: {base}")
        else:
            figure_manifest = figures / "analysis_source_manifest.json"
            _nonempty(figure_manifest)
            if _json(figure_manifest).get("complete") is not True:
                raise ValueError(f"Requested figures are not complete: {figures}")
            required.append(figure_manifest)

    return {str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in required}
