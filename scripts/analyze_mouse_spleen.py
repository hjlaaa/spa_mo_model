#!/usr/bin/env python3
"""Analyze spa_mo_model Mouse Spleen embeddings."""

from __future__ import annotations

import sys
from pathlib import Path


ORIGINAL = Path(__file__).with_name("analyze_crc_stereocite_clustering.py")
INPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/results/mouse_spleen/"
    "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
)


def main() -> None:
    source = ORIGINAL.read_text(encoding="utf-8")
    for old, new in [
        ("CRC_003", "Mouse_Spleen1"),
        ("CRC_006", "Mouse_Spleen2"),
        ("CRC Stereo-CITE-seq", "Mouse Spleen"),
        ("CRC_STEREOCITE", "MOUSE_SPLEEN"),
        ("CRC joint KMeans", "Mouse Spleen joint KMeans"),
        ("CRC independent KMeans", "Mouse Spleen independent KMeans"),
        ("CRC training", "Mouse Spleen training"),
    ]:
        source = source.replace(old, new)
    namespace = {"__file__": str(ORIGINAL), "__name__": "mouse_spleen_spa_mo_analysis"}
    exec(compile(source, str(ORIGINAL), "exec"), namespace)
    sys.argv[1:1] = [
        "--input_dir", str(INPUT_DIR),
        "--output_dir", str(INPUT_DIR / "clustering_analysis"),
        "--n_clusters", "5,8,10,12",
        "--seed", "42",
        "--point_size", "10",
    ]
    namespace["main"]()


if __name__ == "__main__":
    main()
