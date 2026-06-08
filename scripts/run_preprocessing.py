#!/usr/bin/env python3
"""Command-line entry for COSIE-style preprocessing construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.configure import get_default_preprocess_config
from model.multimodal_preprocessing import (
    preprocess_multisection_cosie_style,
    summarize_data_dict,
    summarize_feature_dict,
    summarize_spatial_loc_dict,
)
from model.utils import ensure_dir


def _none_if_empty(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "na"}:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build COSIE-style multimodal preprocessing inputs.",
    )
    parser.add_argument("--config", default=None, help="JSON config for multi-section input.")
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--stage_id", default=None)
    parser.add_argument("--he_input", default=None)
    parser.add_argument("--he_mask_input", default=None)
    parser.add_argument("--he_feature_input", default=None)
    parser.add_argument("--he_reference_adata_input", default=None)
    parser.add_argument("--rna_input", default=None)
    parser.add_argument("--protein_input", default=None)
    parser.add_argument("--metabolite_input", default=None)
    parser.add_argument("--spatial_key", default="spatial")
    parser.add_argument("--uni_feature_key", default="UNI_feature")
    parser.add_argument("--output_summary", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_sections_from_args(args):
    section_id = args.stage_id or args.sample_id or "s1"
    return [
        {
            "section_id": section_id,
            "he_input": _none_if_empty(args.he_input),
            "he_mask_input": _none_if_empty(args.he_mask_input),
            "he_feature_input": _none_if_empty(args.he_feature_input),
            "he_reference_adata_input": _none_if_empty(args.he_reference_adata_input),
            "rna_input": _none_if_empty(args.rna_input),
            "protein_input": _none_if_empty(args.protein_input),
            "metabolite_input": _none_if_empty(args.metabolite_input),
            "spatial_key": args.spatial_key,
            "uni_feature_key": args.uni_feature_key,
        }
    ]


def build_run_config(args, config_data):
    cfg = get_default_preprocess_config()
    if config_data:
        for key, value in config_data.items():
            if key in {"paths", "preprocessing", "he_image"} and isinstance(value, dict):
                cfg[key].update(value)
            elif key != "sections":
                cfg[key] = value

    nested_preprocessing = config_data.get("preprocessing", {}) if config_data else {}
    cfg["preprocessing"]["spatial_key"] = (
        config_data.get("spatial_key", nested_preprocessing.get("spatial_key", args.spatial_key))
        if config_data
        else args.spatial_key
    )
    cfg["preprocessing"]["uni_feature_key"] = (
        config_data.get(
            "uni_feature_key",
            nested_preprocessing.get("uni_feature_key", args.uni_feature_key),
        )
        if config_data
        else args.uni_feature_key
    )
    return cfg


def build_summary(result, dry_run):
    return {
        "dry_run": bool(dry_run),
        "section_ids": result["section_ids"],
        "modality_present": result["modality_present"],
        "data_dict": summarize_data_dict(result["data_dict"]),
        "feature_dict": summarize_feature_dict(result["feature_dict"]),
        "spatial_loc_dict": summarize_spatial_loc_dict(result["spatial_loc_dict"]),
        "messages": result["messages"],
    }


def main():
    args = parse_args()
    config_data = load_config(args.config) if args.config else {}
    sections = config_data.get("sections") if config_data else build_sections_from_args(args)
    if not sections:
        raise ValueError("No sections were provided.")

    run_config = build_run_config(args, config_data)
    preprocessing_cfg = run_config["preprocessing"]

    result = preprocess_multisection_cosie_style(
        sections,
        n_comps=config_data.get("n_comps", preprocessing_cfg["n_comps"]) if config_data else preprocessing_cfg["n_comps"],
        hvg_num=config_data.get("hvg_num", preprocessing_cfg["hvg_num"]) if config_data else preprocessing_cfg["hvg_num"],
        target_sum=config_data.get("target_sum", preprocessing_cfg["target_sum"]) if config_data else preprocessing_cfg["target_sum"],
        use_harmony=config_data.get("use_harmony", preprocessing_cfg["use_harmony"]) if config_data else preprocessing_cfg["use_harmony"],
        metacell=config_data.get("metacell", preprocessing_cfg["metacell"]) if config_data else preprocessing_cfg["metacell"],
        config=run_config,
        dry_run=args.dry_run,
    )

    summary = build_summary(result, dry_run=args.dry_run)
    summary_text = json.dumps(summary, indent=2)
    print(summary_text)

    if args.output_summary:
        ensure_dir(args.output_summary)
        with open(args.output_summary, "w", encoding="utf-8") as handle:
            handle.write(summary_text)
            handle.write("\n")
        print(f"Summary saved to {args.output_summary}")


if __name__ == "__main__":
    main()
