#!/usr/bin/env python3
"""Analyze spa_mo_model Mouse Thymus embeddings with section-specific spot sizes."""

from __future__ import annotations

import sys
from pathlib import Path


ORIGINAL = Path(__file__).with_name("analyze_misar_seq_clustering.py")
RUN_DIR = Path(
    "/home/hujinlan/spa_mo_model/results/mouse_thymus/"
    "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
)
SPOT_SIZES = {
    "Mouse_Thymus1": 8.0,
    "Mouse_Thymus2": 5.0,
    "Mouse_Thymus3": 5.0,
    "Mouse_Thymus4": 5.0,
}


def main() -> None:
    source = ORIGINAL.read_text(encoding="utf-8").replace("MISAR-seq", "Mouse Thymus")
    namespace = {"__file__": str(ORIGINAL), "__name__": "mouse_thymus_spa_mo_analysis"}
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    namespace["DEFAULT_SECTION_ORDER"] = [f"Mouse_Thymus{i}" for i in range(1, 5)]
    namespace["DEFAULT_LABEL_KEYS"] = []
    original_plot_spatial = namespace["plot_spatial"]

    def plot_spatial(
        coords, labels, title, path, point_size, dpi, plot_max_points, seed
    ):
        plot_text = f"{path} {title}"
        section_size = next(
            (size for section, size in SPOT_SIZES.items() if section in plot_text),
            point_size,
        )
        return original_plot_spatial(
            coords,
            labels,
            title,
            path,
            section_size,
            dpi,
            plot_max_points,
            seed,
        )

    namespace["plot_spatial"] = plot_spatial
    sys.argv[1:1] = [
        "--input_dir", str(RUN_DIR),
        "--output_dir", str(RUN_DIR / "clustering_analysis"),
        "--section_order", ",".join(namespace["DEFAULT_SECTION_ORDER"]),
        "--n_clusters", "5,8,10,12",
        "--label_keys", "section",
        "--seed", "42",
        "--point_size", "5",
        "--kmeans_method", "kmeans",
        "--n_init", "20",
        "--metric_sample_size", "10000",
        "--batch_metrics_max_samples", "0",
    ]
    namespace["main"]()
    with (RUN_DIR / "clustering_analysis" / "SUMMARY.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "\n## Mouse Thymus plot adaptation\n\n"
            "- Clustering spot sizes: Mouse_Thymus1 = 8; "
            "Mouse_Thymus2/3/4 = 5.\n"
        )


if __name__ == "__main__":
    main()
