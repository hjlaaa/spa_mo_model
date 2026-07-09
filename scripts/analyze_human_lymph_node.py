#!/usr/bin/env python3
"""Analyze spa_mo_model Human Lymph Node embeddings."""

from __future__ import annotations

import sys
from pathlib import Path


ORIGINAL = Path(__file__).with_name("analyze_crc_stereocite_clustering.py")
INPUT_DIR = Path(
    "/home/hujinlan/spa_mo_model/results/human_lymph_node/"
    "fullspot_200ep_bidirectional_all_checkpoint_lc0.1_seed42"
)


def main() -> None:
    source = ORIGINAL.read_text(encoding="utf-8")
    for old, new in [
        ("CRC_003", "Human_Lymph_Node_A1"),
        ("CRC_006", "Human_Lymph_Node_D1"),
        ("CRC Stereo-CITE-seq", "Human Lymph Node"),
        ("CRC_STEREOCITE", "HUMAN_LYMPH_NODE"),
        ("CRC joint KMeans", "Human Lymph Node joint KMeans"),
        ("CRC independent KMeans", "Human Lymph Node independent KMeans"),
        ("CRC training", "Human Lymph Node training"),
    ]:
        source = source.replace(old, new)
    namespace = {
        "__file__": str(ORIGINAL),
        "__name__": "human_lymph_node_spa_mo_analysis",
    }
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
