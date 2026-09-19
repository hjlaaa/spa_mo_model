"""UMAP computation and saved-coordinate rendering; explicit, independent output."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from typing import Any
import hashlib
import json
import math
import numpy as np
import pandas as pd
import umap
from analysis.cache import (array_identity, frame_identity, file_identity, summary_identity,
    cache_hit, save_cache, begin_analysis, finish_analysis, check_output_path,
    implementation_identity)
from analysis.plotting import (plot_panel_c, plot_panel_e, plot_individual,
    has_meaningful_biological_labels)
from analysis.protocols import umap_parameters

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_features(name: str, values: np.ndarray, n_obs: int) -> None:
    if values.ndim != 2 or len(values) != n_obs:
        raise ValueError(f"{name}: invalid shape {values.shape}; expected {n_obs} rows.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite values detected.")


def human_celltype_colors(bundle: Any) -> dict[str, Any] | None:
    if bundle.name != "human_embryo":
        return None
    path = bundle.run_dir / "analysis/celltype_reference/celltype_color_map.csv"
    table = pd.read_csv(path)
    return dict(zip(table["celltype"].astype(str), table["color"].astype(str)))


def fit_umap(values, cache_path: Path, n_neighbors: int, min_dist: float,
             seed: int, overwrite: bool, *, source_identity: dict,
             force_float32: bool = True) -> np.ndarray:
    """Preserve D8c's float32 input and D8b's supplied input dtype explicitly."""
    cache_path = check_output_path(cache_path)
    parameters = umap_parameters(n_neighbors, min_dist, seed)
    identity = {"source": source_identity, "input": array_identity(values),
                "force_float32": force_float32, "parameters": parameters,
                "implementation": implementation_identity("umap-runtime-v1"),
                "umap_version": umap.__version__}
    manifest = cache_path.with_suffix(".source.json")
    hit = cache_hit(manifest, identity, [cache_path])
    if hit and not overwrite:
        return np.load(cache_path)
    reducer = umap.UMAP(**parameters)
    inputs = np.asarray(values, dtype=np.float32) if force_float32 else values
    coordinates = np.asarray(reducer.fit_transform(inputs), dtype=np.float32)
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid UMAP coordinates: {coordinates.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, coordinates)
    save_cache(manifest, identity, [cache_path])
    return coordinates


def bundle_identity(bundle, args) -> dict:
    return {"dataset": bundle.name, "section_order": list(bundle.sections),
            "metadata": frame_identity(bundle.metadata),
            "input_features": {key: array_identity(value) for key, value in bundle.input_features.items()},
            "integrated_features": array_identity(bundle.integrated_features),
            "selections": {key: array_identity(value) for key, value in bundle.selections.items()},
            "joint_labels": {str(k): array_identity(v) for k,v in bundle.joint_labels.items()},
            "representative_k": bundle.representative_k, "primary_label": bundle.primary_label,
            "run": summary_identity(bundle.run_dir), "colors": human_celltype_colors(bundle),
            "protocol": umap_parameters(args.n_neighbors, args.min_dist, args.seed),
            "implementation": implementation_identity("spamosaic-umap-panels-v1")}


def generate_dataset(bundle: Any, args: Any, *, output_dir: Path) -> None:
    output_dir = check_output_path(output_dir, [bundle.run_dir, bundle.analysis_dir])
    source = bundle_identity(bundle, args)
    if begin_analysis(output_dir, source, input_dirs=[bundle.run_dir, bundle.analysis_dir]):
        return
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    n_obs = len(bundle.metadata)
    point_size = 2.0 if n_obs >= 50_000 else 8.0
    input_coordinates = {}
    for modality, values in bundle.input_features.items():
        validate_features(f"{bundle.name} input {modality}", values, n_obs)
        input_coordinates[modality] = fit_umap(
            values,
            tables_dir / f"input_{modality.lower()}_umap.npy",
            args.n_neighbors,
            args.min_dist,
            args.seed,
            args.overwrite_umap, source_identity=source,
        )
    validate_features(f"{bundle.name} integrated", bundle.integrated_features, n_obs)
    integrated_coordinates = fit_umap(
        bundle.integrated_features,
        tables_dir / "integrated_embedding_umap.npy",
        args.n_neighbors,
        args.min_dist,
        args.seed,
        args.overwrite_umap, source_identity=source,
    )
    custom_biology = human_celltype_colors(bundle)
    include_biology = has_meaningful_biological_labels(bundle)
    plot_panel_c(bundle, input_coordinates, figures_dir, point_size, args.seed)
    plot_panel_e(
        bundle, integrated_coordinates, figures_dir, point_size, args.seed,
        custom_biology,
    )

    section_values = bundle.metadata["section_label"].to_numpy()
    biology_values = bundle.metadata[bundle.primary_label].to_numpy()
    for offset, (modality, coordinates) in enumerate(input_coordinates.items()):
        plot_individual(
            figures_dir / f"input_{modality.lower()}_by_section.png",
            coordinates, section_values,
            f"{bundle.display_name} input {modality} UMAP by section",
            "section", point_size, args.seed + offset,
        )
        biological_path = (
            figures_dir / f"input_{modality.lower()}_by_{bundle.primary_label}.png"
        )
        if include_biology:
            plot_individual(
                biological_path, coordinates, biology_values,
                f"{bundle.display_name} input {modality} UMAP by {bundle.primary_label}",
                "biology", point_size, args.seed + offset, custom_biology,
            )
        else:
            biological_path.unlink(missing_ok=True)
    plot_individual(
        figures_dir / "integrated_by_section.png",
        integrated_coordinates, section_values,
        f"{bundle.display_name} integrated embedding UMAP by section",
        "section", point_size, args.seed,
    )
    integrated_biological_path = (
        figures_dir / f"integrated_by_{bundle.primary_label}.png"
    )
    if include_biology:
        plot_individual(
            integrated_biological_path, integrated_coordinates, biology_values,
            f"{bundle.display_name} integrated embedding UMAP by {bundle.primary_label}",
            "biology", point_size, args.seed, custom_biology,
        )
    else:
        integrated_biological_path.unlink(missing_ok=True)
    cluster_dir = figures_dir / "integrated_joint_clusters"
    for k, labels in sorted(bundle.joint_labels.items()):
        plot_individual(
            cluster_dir / f"integrated_by_joint_k{k}.png",
            integrated_coordinates, labels.astype(str),
            f"{bundle.display_name} integrated embedding UMAP by joint k={k}",
            "cluster", point_size, args.seed,
        )

    coordinates_table = bundle.metadata.copy()
    for modality, coordinates in input_coordinates.items():
        key = modality.lower()
        coordinates_table[f"input_{key}_UMAP1"] = coordinates[:, 0]
        coordinates_table[f"input_{key}_UMAP2"] = coordinates[:, 1]
    coordinates_table["integrated_UMAP1"] = integrated_coordinates[:, 0]
    coordinates_table["integrated_UMAP2"] = integrated_coordinates[:, 1]
    for k, labels in sorted(bundle.joint_labels.items()):
        coordinates_table[f"joint_k{k}"] = labels
    coordinate_path = tables_dir / "umap_coordinates_and_labels.csv.gz"
    coordinates_table.to_csv(coordinate_path, index=False, compression="gzip")
    sample_path = tables_dir / "sample_indices_by_section.npz"
    np.savez_compressed(sample_path, **bundle.selections)

    config = {
        "status": "PASS",
        "dataset": bundle.name,
        "display_name": bundle.display_name,
        "run_dir": bundle.run_dir,
        "analysis_dir": bundle.analysis_dir,
        "output_dir": output_dir,
        "panel_mapping": {
            "c": (
                "input-modality UMAPs colored by section and primary biological label"
                if include_biology else
                "input-modality UMAPs colored by section; biological annotation omitted because unavailable"
            ),
            "e": (
                "integrated final-embedding UMAP colored by section, representative joint cluster, and primary biological label"
                if include_biology else
                "integrated final-embedding UMAP colored by section and representative joint cluster; biological annotation omitted because unavailable"
            ),
        },
        "n_obs_total": sum(bundle.section_counts.values()),
        "n_obs_umap": n_obs,
        "sampled": n_obs < sum(bundle.section_counts.values()),
        "sampling": "section-proportional uniform without replacement",
        "section_counts_total": bundle.section_counts,
        "section_counts_umap": {s: len(bundle.selections[s]) for s in bundle.sections},
        "section_order": bundle.sections,
        "primary_biological_label": bundle.primary_label,
        "primary_biological_label_available": include_biology,
        "input_modalities": list(bundle.input_features),
        "input_provenance": bundle.input_provenance,
        "integrated_input": "joint StandardScaler-transformed saved final embedding",
        "umap": {
            "implementation": "umap-learn",
            "version": umap.__version__,
            "n_components": 2,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "euclidean",
            "init": "spectral",
            "random_state": args.seed,
            "transform_seed": args.seed,
            "n_jobs": 1,
            "low_memory": True,
        },
        "joint_cluster_k_values": sorted(bundle.joint_labels),
        "representative_joint_k": bundle.representative_k,
        "representative_k_rule": "existing best/nearest retained biological-label result; MouseBrain uses the established retained k=10",
        "coordinates_table": coordinate_path,
        "sample_indices": sample_path,
        "figures": {
            "panel_c": figures_dir / "panel_c_input_modality_umap.png",
            "panel_e": figures_dir / "panel_e_integrated_embedding_umap.png",
        },
        "created_at": datetime.now().astimezone().isoformat(),
        "script": Path(__file__).resolve(),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output_dir / "umap_config.json", config)
    readme = [
        f"# {bundle.display_name} SpaMosaic-style UMAP", "",
        "- Panel c: model-input modality representations before spa_mo_model integration.",
        "- Panel e: saved final embedding after spa_mo_model integration.",
        f"- UMAP spots: {n_obs:,} of {sum(bundle.section_counts.values()):,}.",
        (
            f"- Primary biological label: `{bundle.primary_label}`."
            if include_biology else
            "- Primary biological annotation is unavailable; annotation-colored UMAPs are intentionally omitted."
        ),
        f"- Representative joint clustering: k={bundle.representative_k}.",
        "- UMAP is a qualitative visualization; integration claims should also use the saved quantitative metrics.",
        "",
        "All before/after panels use the exact same spot identities. Large datasets use a deterministic section-proportional sample.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    finish_analysis(output_dir, source)
    print(f"[umap] completed {bundle.name}: {output_dir}", flush=True)


def color_map(analysis_dir: Path, primary_label: str) -> dict[str, str] | None:
    if primary_label != "celltype":
        return None
    path = analysis_dir / "celltype_reference/celltype_color_map.csv"
    if not path.is_file():
        return None
    table = pd.read_csv(path)
    return dict(zip(table["celltype"].astype(str), table["color"].astype(str)))


def rerender(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text())
    table = pd.read_csv(config["coordinates_table"], low_memory=False)
    output_dir = check_output_path(output_dir, [config_path.parent])
    source = {"coordinates": file_identity(Path(config["coordinates_table"])),
              "metadata": frame_identity(table),
              "protocol": {key: config[key] for key in ("dataset", "display_name", "input_modalities", "primary_biological_label", "representative_joint_k", "umap")},
              "implementation": implementation_identity("saved-umap-render-v1")}
    colors = color_map(Path(config["analysis_dir"]), config["primary_biological_label"])
    source["colors"] = colors
    if begin_analysis(output_dir, source, input_dirs=[config_path.parent]):
        return
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    modalities = list(config["input_modalities"])
    input_coordinates = {
        modality: table[
            [f"input_{modality.lower()}_UMAP1", f"input_{modality.lower()}_UMAP2"]
        ].to_numpy(dtype=np.float32)
        for modality in modalities
    }
    integrated = table[
        ["integrated_UMAP1", "integrated_UMAP2"]
    ].to_numpy(dtype=np.float32)
    joint_labels = {
        int(column.removeprefix("joint_k")): table[column].to_numpy(dtype=np.int32)
        for column in table.columns
        if column.startswith("joint_k")
    }
    bundle = SimpleNamespace(
        name=config["dataset"],
        display_name=config["display_name"],
        analysis_dir=Path(config["analysis_dir"]),
        metadata=table,
        primary_label=config["primary_biological_label"],
        representative_k=int(config["representative_joint_k"]),
        joint_labels=joint_labels,
    )
    point_size = 2.0 if len(table) >= 50_000 else 8.0
    seed = int(config["umap"]["random_state"])
    custom_biology = color_map(bundle.analysis_dir, bundle.primary_label)
    include_biology = has_meaningful_biological_labels(bundle)

    if input_coordinates:
        plot_panel_c(bundle, input_coordinates, figures_dir, point_size, seed)
    plot_panel_e(
        bundle, integrated, figures_dir, point_size, seed, custom_biology
    )

    section_values = table["section_label"].to_numpy()
    biology_values = table[bundle.primary_label].to_numpy()
    for offset, (modality, coordinates) in enumerate(input_coordinates.items()):
        plot_individual(
            figures_dir / f"input_{modality.lower()}_by_section.png",
            coordinates,
            section_values,
            f"{bundle.display_name} input {modality} UMAP by section",
            "section",
            point_size,
            seed + offset,
        )
        biology_path = (
            figures_dir / f"input_{modality.lower()}_by_{bundle.primary_label}.png"
        )
        if include_biology:
            plot_individual(
                biology_path,
                coordinates,
                biology_values,
                f"{bundle.display_name} input {modality} UMAP by {bundle.primary_label}",
                "biology",
                point_size,
                seed + offset,
                custom_biology,
            )
        else:
            biology_path.unlink(missing_ok=True)

    plot_individual(
        figures_dir / "integrated_by_section.png",
        integrated,
        section_values,
        f"{bundle.display_name} integrated embedding UMAP by section",
        "section",
        point_size,
        seed,
    )
    integrated_biology_path = (
        figures_dir / f"integrated_by_{bundle.primary_label}.png"
    )
    if include_biology:
        plot_individual(
            integrated_biology_path,
            integrated,
            biology_values,
            f"{bundle.display_name} integrated embedding UMAP by {bundle.primary_label}",
            "biology",
            point_size,
            seed,
            custom_biology,
        )
    else:
        integrated_biology_path.unlink(missing_ok=True)
    cluster_dir = figures_dir / "integrated_joint_clusters"
    for k, labels in sorted(joint_labels.items()):
        plot_individual(
            cluster_dir / f"integrated_by_joint_k{k}.png",
            integrated,
            labels.astype(str),
            f"{bundle.display_name} integrated embedding UMAP by joint k={k}",
            "cluster",
            point_size,
            seed,
        )

    config["figure_layout"] = {
        "title": "centered; panel-letter prefixes omitted",
        "multi_panel_legend": "below each subplot",
        "single_panel_legend": (
            "right for at most 10 short labels; below for many or long labels"
        ),
        "coordinates_refit": False,
    }
    config["figure_render"] = {
        "script": Path(__file__).resolve(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "rendered_at": datetime.now().astimezone().isoformat(),
    }
    config["output_dir"] = str(output_dir)
    write_json(output_dir / "umap_config.json", config)
    finish_analysis(output_dir, source)
    print(f"[rerender-umap] completed {config['dataset']}: {figures_dir}", flush=True)
