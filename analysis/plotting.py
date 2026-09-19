"""Spatial, annotation and UMAP renderers. Numerical results are supplied by callers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pandas as pd

SECTION_COLORS = list(plt.get_cmap("tab10").colors)
MISSING_COLOR = "#d3d3d3"

def plot_spatial_categories(
    path: Path, coords: np.ndarray, labels: np.ndarray, title: str, *, point_size: float, dpi: int
) -> None:
    categories, codes = np.unique(labels, return_inverse=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        s=point_size,
        cmap="tab20",
        linewidths=0,
        alpha=0.95,
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    handles, _ = scatter.legend_elements(num=len(categories))
    ax.legend(
        handles,
        [str(item) for item in categories],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_crc_spatial(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=0.45,
        cmap=plt.get_cmap("tab20", k),
        vmin=-0.5,
        vmax=k - 0.5,
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, ticks=np.arange(k))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatch_spatial(
    path: Path,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    title: str,
    k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(
        metadata["x"],
        metadata["y"],
        c=labels,
        s=0.2,
        cmap=plt.get_cmap("tab20", min(k, 20)),
        vmin=-0.5,
        vmax=k - 0.5,
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axis("equal")
    ax.axis("off")
    fig.colorbar(scatter, ax=ax, ticks=np.arange(k))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def categorical_colors(n_categories: int) -> list[tuple[float, float, float, float]]:
    colors: list[tuple[float, float, float, float]] = []
    for name in ("tab20", "tab20b", "tab20c"):
        colors.extend(list(plt.get_cmap(name).colors))
    if n_categories > len(colors):
        colors.extend([plt.get_cmap("hsv")(value) for value in np.linspace(0, 1, n_categories - len(colors), endpoint=False)])
    return colors[:n_categories]


def plot_hierarchy_tree(path: Path, linkage: np.ndarray, title: str, n_leaves: int) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage, labels=[f"leaf {leaf}" for leaf in range(n_leaves)],
        leaf_rotation=45, leaf_font_size=8, color_threshold=0, above_threshold_color="#333333",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel(f"flat k={n_leaves} leaf cluster")
    axis.set_ylabel("sqrt(weighted Ward SSE increase)")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_panels(
    path: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    labels: dict[str, np.ndarray],
    title: str,
    maximum: int,
    seed: int,
    k: int,
    *, selections: dict[str, np.ndarray]
) -> None:
    if k < 2 or k > 40:
        raise ValueError(f"The discrete cluster plot requires 2 <= k <= 40; got {k}.")
    colors = list(plt.get_cmap("tab20").colors)
    if k > len(colors):
        colors.extend(list(plt.get_cmap("tab20b").colors[: k - len(colors)]))
    colors = colors[:k]
    cmap = ListedColormap(colors, name=f"tab20_family_{k}_discrete")
    boundaries = np.arange(-0.5, k + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), squeeze=False)
    for offset, section in enumerate(sections):
        axis = axes.ravel()[offset]
        frame = metadata[section]
        chosen = selections[section]
        values = np.asarray(labels[section][chosen], dtype=np.int32)
        if values.size and (values.min() < 0 or values.max() >= k):
            raise ValueError(f"{section}: cluster labels fall outside [0, {k - 1}].")
        axis.scatter(
            frame["x"].to_numpy()[chosen], frame["y"].to_numpy()[chosen],
            c=values, s=1, cmap=cmap, norm=norm, linewidths=0, rasterized=True,
        )
        axis.set_title(section)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
    for axis in axes.ravel()[len(sections):]:
        axis.axis("off")
    fig.suptitle(title)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(
        scalar, ax=axes.ravel().tolist(), shrink=0.55,
        ticks=np.arange(k), boundaries=boundaries, spacing="uniform", label="cluster",
    )
    colorbar.ax.set_yticklabels([str(value) for value in range(k)])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_celltype_reference(
    output_dir: Path,
    sections: list[str],
    metadata: dict[str, pd.DataFrame],
    maximum: int,
    seed: int,
    *, selections: dict[str, np.ndarray]
) -> dict[str, Any]:
    reference_dir = output_dir / "celltype_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    categories = sorted(
        {
            value
            for section in sections
            for value in metadata[section]["celltype"].astype(str).tolist()
            if value != "nan"
        }
    )
    if not categories:
        raise ValueError("No valid celltype annotations were found.")
    colors = categorical_colors(len(categories))
    category_index = {name: index for index, name in enumerate(categories)}
    fig, axes = plt.subplots(3, 3, figsize=(20, 15), squeeze=False)
    count_rows: list[dict[str, Any]] = []
    for offset, section in enumerate(sections):
        axis = axes.ravel()[offset]
        frame = metadata[section]
        truth = frame["celltype"].astype(str).to_numpy()
        chosen = selections[section]
        valid = truth[chosen] != "nan"
        chosen_valid = chosen[valid]
        point_colors = [colors[category_index[value]] for value in truth[chosen_valid]]
        axis.scatter(
            frame["x"].to_numpy()[chosen_valid], frame["y"].to_numpy()[chosen_valid],
            c=point_colors, s=1, linewidths=0, rasterized=True,
        )
        axis.set_title(section)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
        for celltype, count in frame["celltype"].astype(str).value_counts().items():
            if celltype != "nan":
                count_rows.append(
                    {
                        "section": section,
                        "celltype": celltype,
                        "count": int(count),
                        "fraction": float(count / len(frame)),
                        "color": to_hex(colors[category_index[celltype]]),
                    }
                )
    for axis in axes.ravel()[len(sections):]:
        axis.axis("off")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5, color=colors[index], label=name)
        for index, name in enumerate(categories)
    ]
    fig.suptitle(f"celltype reference annotations ({len(categories)} classes)")
    fig.legend(
        handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.5),
        ncol=2, fontsize=7, title="celltype", markerscale=1.2,
    )
    fig.subplots_adjust(right=0.72)
    figure_path = reference_dir / "spatial_all_sections_by_celltype.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(reference_dir / "celltype_counts_by_section.csv", index=False)
    color_map = pd.DataFrame(
        {"celltype": categories, "color": [to_hex(color) for color in colors]}
    )
    color_map.to_csv(reference_dir / "celltype_color_map.csv", index=False)
    return {
        "n_celltypes": len(categories),
        "figure": str(figure_path),
        "counts": str(reference_dir / "celltype_counts_by_section.csv"),
        "color_map": str(reference_dir / "celltype_color_map.csv"),
    }


def categorical_palette(count: int) -> list[Any]:
    colors: list[Any] = []
    for name in ("tab20", "tab20b", "tab20c"):
        colors.extend(plt.get_cmap(name).colors)
    if count > len(colors):
        colors.extend(
            plt.get_cmap("hsv")(
                np.linspace(0, 1, count - len(colors), endpoint=False)
            )
        )
    return colors[:count]


def natural_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def clean_labels(values: Any) -> np.ndarray:
    series = pd.Series(values, dtype=object)
    result = series.astype(str).to_numpy(dtype=object)
    missing = series.isna().to_numpy() | np.isin(
        np.char.lower(result.astype(str)), ["nan", "none", "na", ""]
    )
    result[missing] = "NA"
    return result


def color_lookup(
    values: np.ndarray,
    kind: str,
    custom: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    cleaned = clean_labels(values)
    categories = sorted(
        [value for value in np.unique(cleaned).tolist() if value != "NA"],
        key=natural_key,
    )
    if kind == "section":
        colors = SECTION_COLORS[: len(categories)]
    else:
        colors = categorical_palette(len(categories))
    lookup = dict(zip(categories, colors))
    if custom:
        lookup.update({str(key): value for key, value in custom.items() if str(key) in categories})
    if "NA" in cleaned:
        categories.append("NA")
        lookup["NA"] = MISSING_COLOR
    return categories, lookup


def scatter_panel(
    axis: Any,
    coordinates: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    lookup: dict[str, Any],
    title: str,
    point_size: float,
    seed: int,
    legend_below: bool = True,
    legend_ncol: int | None = None,
) -> None:
    cleaned = clean_labels(labels)
    order = np.random.default_rng(seed).permutation(len(coordinates))
    axis.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=[lookup[value] for value in cleaned[order]],
        s=point_size,
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    axis.set_title(title, fontsize=10)
    axis.set_aspect("equal", adjustable="datalim")
    axis.axis("off")
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=4,
            color=lookup[value], label=str(value),
        )
        for value in categories
    ]
    ncol = (
        legend_ncol
        if legend_ncol is not None
        else max(1, min(6, math.ceil(len(categories) / 5)))
    )
    if legend_below:
        axis.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            frameon=False,
            fontsize=6,
            ncol=ncol,
            handletextpad=0.25,
            columnspacing=0.6,
        )
    else:
        axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=False,
            fontsize=7,
            ncol=ncol,
        )


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_individual(
    path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    kind: str,
    point_size: float,
    seed: int,
    custom: dict[str, Any] | None = None,
) -> None:
    categories, lookup = color_lookup(labels, kind, custom)
    maximum_label_length = max((len(str(value)) for value in categories), default=0)
    legend_on_right = len(categories) <= 10 and maximum_label_length <= 24
    if legend_on_right:
        figure, axis = plt.subplots(figsize=(8.2, 5.8))
        scatter_panel(
            axis,
            coordinates,
            labels,
            categories,
            lookup,
            "",
            point_size,
            seed,
            False,
            1,
        )
        figure.subplots_adjust(left=0.05, right=0.75, bottom=0.06, top=0.88)
    else:
        if maximum_label_length > 24:
            legend_ncol = min(4, max(1, len(categories)))
            figure_width = 10.5
        elif maximum_label_length > 14:
            legend_ncol = min(5, max(1, len(categories)))
            figure_width = 9.2
        else:
            legend_ncol = min(8, max(1, len(categories)))
            figure_width = 8.2
        legend_rows = max(1, math.ceil(len(categories) / legend_ncol))
        figure_height = 5.8 + min(2.2, 0.25 * legend_rows)
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        scatter_panel(
            axis,
            coordinates,
            labels,
            categories,
            lookup,
            "",
            point_size,
            seed,
            True,
            legend_ncol,
        )
        figure.subplots_adjust(
            left=0.04,
            right=0.96,
            bottom=min(0.43, 0.13 + 0.045 * legend_rows),
            top=0.88,
        )
    figure.suptitle(title, x=0.5, ha="center", fontsize=12)
    save_figure(figure, path)


def has_meaningful_biological_labels(bundle: Any) -> bool:
    """Return true only when at least one biological annotation is available."""
    values = pd.Series(bundle.metadata[bundle.primary_label], dtype="object")
    normalized = values.astype(str).str.strip().str.lower()
    missing = normalized.isin(
        {"", "na", "n/a", "nan", "none", "null", "not_available", "not available"}
    )
    return bool((values.notna() & ~missing).any())


def plot_panel_c(
    bundle: Any,
    input_coordinates: dict[str, np.ndarray],
    figures_dir: Path,
    point_size: float,
    seed: int,
) -> None:
    modalities = list(input_coordinates)
    section_values = bundle.metadata["section_label"].to_numpy()
    section_categories, section_lookup = color_lookup(section_values, "section")
    include_biology = has_meaningful_biological_labels(bundle)
    biological_values = bundle.metadata[bundle.primary_label].to_numpy()
    biological_categories, biological_lookup = (
        color_lookup(biological_values, "biology")
        if include_biology else ([], {})
    )
    many_biology_labels = include_biology and len(biological_categories) > 30
    columns_per_modality = 2 if include_biology else 1
    figure, axes = plt.subplots(
        1,
        columns_per_modality * len(modalities),
        figsize=(
            max(7.2, 7.2 * len(modalities)),
            7.7 if many_biology_labels else 6.7,
        ),
        squeeze=False,
    )
    for index, modality in enumerate(modalities):
        left = axes[0, columns_per_modality * index]
        scatter_panel(
            left, input_coordinates[modality], section_values,
            section_categories, section_lookup, f"{modality}\nSection label",
            point_size, seed + index,
        )
        if include_biology:
            right = axes[0, columns_per_modality * index + 1]
            scatter_panel(
                right, input_coordinates[modality], biological_values,
                biological_categories, biological_lookup,
                f"{modality}\n{bundle.primary_label} label",
                point_size, seed + index,
                legend_ncol=4 if many_biology_labels else None,
            )
    figure.suptitle(
        f"{bundle.display_name}: input-modality UMAP",
        x=0.5, ha="center", fontsize=12, fontweight="bold",
    )
    figure.subplots_adjust(
        bottom=0.39 if many_biology_labels else 0.27,
        wspace=0.08,
        top=0.88,
    )
    save_figure(figure, figures_dir / "panel_c_input_modality_umap.png")


def plot_panel_e(
    bundle: Any,
    integrated_coordinates: np.ndarray,
    figures_dir: Path,
    point_size: float,
    seed: int,
    custom_biology: dict[str, Any] | None,
) -> None:
    k = bundle.representative_k
    if k not in bundle.joint_labels:
        raise KeyError(f"{bundle.name}: representative k={k} labels are unavailable.")
    section_values = bundle.metadata["section_label"].to_numpy()
    cluster_values = bundle.joint_labels[k].astype(str)
    biological_values = bundle.metadata[bundle.primary_label].to_numpy()
    sets = [
        (section_values, "section", "Section label", None),
        (cluster_values, "cluster", f"Joint cluster (k={k})", None),
    ]
    if has_meaningful_biological_labels(bundle):
        sets.append(
            (biological_values, "biology", f"{bundle.primary_label} label", custom_biology)
        )
    figure, axes = plt.subplots(
        1, len(sets), figsize=(6 * len(sets), 6.7), squeeze=False
    )
    for index, (labels, kind, title, custom) in enumerate(sets):
        categories, lookup = color_lookup(labels, kind, custom)
        scatter_panel(
            axes[0, index], integrated_coordinates, labels, categories, lookup,
            title, point_size, seed + index,
        )
    figure.suptitle(
        f"{bundle.display_name}: integrated embedding UMAP",
        x=0.5, ha="center", fontsize=12, fontweight="bold",
    )
    figure.subplots_adjust(bottom=0.27, wspace=0.08, top=0.88)
    save_figure(figure, figures_dir / "panel_e_integrated_embedding_umap.png")


MISSING_BIOLOGY_LABELS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not_available",
    "not available",
    "unknown",
    "unannotated",
}

def meaningful_biology_mask(reference: Any) -> np.ndarray:
    """Identify spots carrying an actual primary biological annotation."""
    values = pd.Series(reference.metadata[reference.primary_label], dtype="object")
    normalized = values.astype(str).str.strip().str.lower()
    return (values.notna() & ~normalized.isin(MISSING_BIOLOGY_LABELS)).to_numpy()

def plot_comparison_panel_e(
    data: Any,
    reference: Any,
    coordinates: np.ndarray,
    figures_dir: Path,
    seed: int,
    *, method_name: str, custom_biology: dict | None,
) -> None:
    section_values = reference.metadata["section_label"].astype(str).to_numpy()
    biological_values = reference.metadata[reference.primary_label].to_numpy()
    cluster_values = data.joint_labels[reference.representative_k].astype(str)
    sets = [
        (section_values, "section", "Section label", None),
        (
            cluster_values,
            "cluster",
            f"Joint cluster (k={reference.representative_k})",
            None,
        ),
    ]
    if meaningful_biology_mask(reference).any():
        sets.append(
            (
                biological_values,
                "biology",
                f"{reference.primary_label} label",
                custom_biology,
            )
        )
    point_size = 2.0 if len(coordinates) >= 50_000 else 8.0
    figure, axes = plt.subplots(
        1, len(sets), figsize=(6 * len(sets), 6.7), squeeze=False
    )
    for index, (labels, kind, title, custom) in enumerate(sets):
        categories, lookup = color_lookup(labels, kind, custom)
        scatter_panel(
            axes[0, index], coordinates, labels, categories, lookup,
            title, point_size, seed + index,
        )
    figure.suptitle(
        f"{method_name} — {reference.config['display_name']}: integrated embedding UMAP",
        x=0.5, ha="center", fontsize=12, fontweight="bold",
    )
    figure.subplots_adjust(bottom=0.27, wspace=0.08, top=0.88)
    save_figure(figure, figures_dir / "panel_e_integrated_embedding_umap.png")

SPATIAL_STYLES = {
    "mousebrain": (10.0, 220), "misar_seq": (18.0, 220),
    "human_lymph_node": (10.0, 220), "mouse_spleen": (10.0, 220),
    "simulation": (28.0, 180),
}
THYMUS_POINT_SIZES = {"Mouse_Thymus1": 8.0, "Mouse_Thymus2": 5.0,
                     "Mouse_Thymus3": 5.0, "Mouse_Thymus4": 5.0}


def plot_dataset_spatial(dataset, path, coords, labels, title, *, section, k):
    if dataset == "crc_stereocite":
        return plot_crc_spatial(path, coords, labels, title, k)
    if dataset == "spatch":
        return plot_spatch_spatial(path, pd.DataFrame(coords, columns=["x", "y"]), labels, title, k)
    point_size, dpi = (THYMUS_POINT_SIZES[section], 220) if dataset == "mouse_thymus" else SPATIAL_STYLES[dataset]
    return plot_spatial_categories(path, coords, labels, title, point_size=point_size, dpi=dpi)


def plot_embryo_section(
    axis: Any,
    frame: pd.DataFrame,
    labels: np.ndarray,
    color_lookup: dict[Any, Any],
    maximum: int,
    seed: int,
    *, chosen: np.ndarray,
) -> None:
    values = labels[chosen]
    axis.scatter(
        frame["x"].to_numpy()[chosen],
        frame["y"].to_numpy()[chosen],
        c=[color_lookup[value] for value in values],
        s=1,
        linewidths=0,
        rasterized=True,
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.axis("off")


def embryo_add_legend(
    figure: Any,
    values: list[Any],
    color_lookup: dict[Any, Any],
    title: str,
    ncol: int = 1,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            color=color_lookup[value],
            label=str(value),
        )
        for value in values
    ]
    figure.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        title=title,
        fontsize=7,
        ncol=ncol,
    )


def save_atomic_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=220, bbox_inches="tight")
    temporary.replace(path)
    plt.close(figure)


def plot_mousebrain_colorbar(path: Path, coords, labels, title: str, n_clusters: int, point_size: float, invert_y: bool, dpi: int):
    cmap = plt.get_cmap("tab20", n_clusters)
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap=cmap,
        s=point_size,
        alpha=0.9,
        linewidths=0,
        vmin=-0.5,
        vmax=n_clusters - 0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label("cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_misar_selected(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
    point_size: float,
    dpi: int,
    *, indices: np.ndarray,
) -> None:
    idx = indices
    labels_subset = labels[idx].astype(str)
    categories, codes = np.unique(labels_subset, return_inverse=True)
    plt.figure(figsize=(5.2, 4.8))
    scatter = plt.scatter(
        coords[idx, 0],
        coords[idx, 1],
        c=codes,
        s=point_size,
        cmap="tab20",
        linewidths=0,
        alpha=0.95,
    )
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    if categories.shape[0] <= 20:
        handles, _ = scatter.legend_elements(num=categories.shape[0])
        plt.legend(
            handles,
            categories.tolist(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=6,
            frameon=False,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_crc_loss_history(output_dir: Path, loss_history: list[dict[str, Any]] | None) -> list[str]:
    if not isinstance(loss_history, list) or not loss_history:
        return []
    loss_dir = output_dir / "training_curves"
    loss_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.array([record.get("epoch") for record in loss_history], dtype=float)
    files = []
    for key in ["total_loss", "crossview_loss", "reconstruction_loss", "weighted_crossview_loss"]:
        values = np.array([record.get(key, np.nan) for record in loss_history], dtype=float)
        if np.all(np.isnan(values)):
            continue
        path = loss_dir / f"{key}.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, values, linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(f"CRC training {key}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))
    return files


def plot_crc_selected(
    path: Path,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    n_clusters: int,
    point_size: float,
    dpi: int,
    *, plot_idx: np.ndarray | slice,
):
    coords_plot = coords[plot_idx]
    labels_plot = labels[plot_idx]
    cmap = plt.get_cmap("tab20", n_clusters)
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        coords_plot[:, 0],
        coords_plot[:, 1],
        c=labels_plot,
        cmap=cmap,
        s=point_size,
        alpha=0.9,
        linewidths=0,
        vmin=-0.5,
        vmax=n_clusters - 0.5,
        rasterized=True,
    )
    suffix = "" if isinstance(plot_idx, slice) else f" ({labels_plot.shape[0]:,} plotted)"
    ax.set_title(title + suffix)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label("cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

