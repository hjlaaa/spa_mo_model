#!/usr/bin/env python3
"""Compare result_v4C with result_v3 at every matched clustering K."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "result_v4C"

DATASETS = {
    "mousebrain": "MouseBrain",
    "human_lymph_node": "Human_Lymph_Node",
    "misar_seq": "MISAR-seq",
    "mouse_spleen": "Mouse_Spleen",
    "mouse_thymus": "Mouse_Thymus",
    "simulation": "Simulation",
    "crc_stereocite": "CRC_Stereo-CITE-seq",
    "spatch": "spatch",
}

TABLES = {
    "mousebrain": [
        "clustering_metrics.csv",
        "clustering_metrics_by_label.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
    "human_lymph_node": [
        "clustering_metrics.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
    "misar_seq": [
        "clustering_metrics.csv",
        "clustering_metrics_by_label.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
    "mouse_spleen": [
        "clustering_metrics.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
    "mouse_thymus": [
        "clustering_metrics.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
    "simulation": [
        "clustering_metrics.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
    "crc_stereocite": [
        "clustering_metrics.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
    "spatch": [
        "external_clustering_metrics.csv",
        "internal_clustering_metrics.csv",
        "spatial_continuity.csv",
        "spatial_continuity_summary.csv",
    ],
}

METRICS = [
    "ari",
    "nmi",
    "homogeneity",
    "completeness",
    "v_measure",
    "label_asw",
    "label_asw_scaled",
    "cluster_asw",
    "cluster_asw_scaled",
    "calinski_harabasz",
    "davies_bouldin",
    "section_ari",
    "section_nmi",
    "section_asw",
    "mean_spatial_neighbor_agreement",
    "neighbor_same_cluster_fraction",
]

LOWER_IS_BETTER = {"davies_bouldin"}
CLOSER_TO_ZERO_IS_BETTER = {"section_ari", "section_nmi", "section_asw"}


def metrics_dir(version: str, dataset: str) -> Path:
    if version == "v3" and dataset == "mousebrain":
        run = "v3_bidirectional_sparse_fixed_lc0.1"
    else:
        run = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
    return ROOT / f"result_{version}" / dataset / run / "analysis" / "standardized_embedding" / "metrics"


def direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in CLOSER_TO_ZERO_IS_BETTER:
        return "closer_to_zero"
    return "higher"


def assess(metric: str, v3: float, v4c: float, tol: float = 1e-12) -> str:
    preferred = direction(metric)
    if preferred == "closer_to_zero":
        change = abs(v3) - abs(v4c)
    elif preferred == "lower":
        change = v3 - v4c
    else:
        change = v4c - v3
    if change > tol:
        return "提升"
    if change < -tol:
        return "下降"
    return "持平"


def compare_table(dataset: str, filename: str) -> list[dict]:
    path_v3 = metrics_dir("v3", dataset) / filename
    path_v4c = metrics_dir("v4C", dataset) / filename
    if not path_v3.exists() or not path_v4c.exists():
        raise FileNotFoundError(f"Missing comparison input: {path_v3} or {path_v4c}")

    v3 = pd.read_csv(path_v3)
    v4c = pd.read_csv(path_v4c)
    key_order = ["mode", "scope", "section", "label", "k"]
    keys = [column for column in key_order if column in v3.columns and column in v4c.columns]
    if "k" not in keys:
        raise ValueError(f"No clustering K column in {filename}")
    if v3.duplicated(keys).any() or v4c.duplicated(keys).any():
        raise ValueError(f"Duplicate comparison keys for {dataset}/{filename}: {keys}")

    merged = v3.merge(v4c, on=keys, how="outer", suffixes=("_v3", "_v4C"), indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", keys + ["_merge"]]
    if not unmatched.empty:
        raise ValueError(
            f"Unmatched rows for {dataset}/{filename}:\n{unmatched.to_string(index=False)}"
        )

    rows: list[dict] = []
    for metric in METRICS:
        left = f"{metric}_v3"
        right = f"{metric}_v4C"
        if left not in merged.columns or right not in merged.columns:
            continue
        for record in merged[keys + [left, right]].to_dict("records"):
            v3_value = record[left]
            v4c_value = record[right]
            if pd.isna(v3_value) or pd.isna(v4c_value):
                continue
            delta = float(v4c_value) - float(v3_value)
            relative = np.nan
            if abs(float(v3_value)) > 1e-12:
                relative = delta / abs(float(v3_value)) * 100.0
            row = {
                "dataset": DATASETS[dataset],
                "dataset_key": dataset,
                "source_file": filename,
                "mode": record.get("mode", ""),
                "scope": record.get("scope", ""),
                "section": record.get("section", ""),
                "label": record.get("label", ""),
                "k": int(record["k"]),
                "metric": metric,
                "preferred_direction": direction(metric),
                "v3": float(v3_value),
                "v4C": float(v4c_value),
                "delta_v4C_minus_v3": delta,
                "relative_change_pct": relative,
                "assessment": assess(metric, float(v3_value), float(v4c_value)),
            }
            rows.append(row)
    return rows


def make_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "dataset_key",
        "source_file",
        "mode",
        "scope",
        "section",
        "label",
        "metric",
        "preferred_direction",
    ]
    summaries = []
    for keys, group in long_df.groupby(group_cols, dropna=False, sort=False):
        group = group.sort_values("k")
        deltas = group["delta_v4C_minus_v3"]
        best = group.loc[deltas.idxmax()]
        worst = group.loc[deltas.idxmin()]
        summaries.append(
            dict(
                zip(group_cols, keys),
                k_values=",".join(str(value) for value in group["k"]),
                common_k_count=len(group),
                improved_k_count=int((group["assessment"] == "提升").sum()),
                declined_k_count=int((group["assessment"] == "下降").sum()),
                equal_k_count=int((group["assessment"] == "持平").sum()),
                improvement_rate=float((group["assessment"] == "提升").mean()),
                mean_v3=float(group["v3"].mean()),
                mean_v4C=float(group["v4C"].mean()),
                mean_delta=float(deltas.mean()),
                median_delta=float(deltas.median()),
                min_delta=float(worst["delta_v4C_minus_v3"]),
                min_delta_k=int(worst["k"]),
                max_delta=float(best["delta_v4C_minus_v3"]),
                max_delta_k=int(best["k"]),
            )
        )
    return pd.DataFrame(summaries)


def format_number(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:+.4f}"


def write_report(long_df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    primary = summary[
        (
            (summary["source_file"].isin(["clustering_metrics.csv", "internal_clustering_metrics.csv"]))
            & (summary["mode"] == "joint")
            & (summary["scope"] == "combined")
            & summary["metric"].isin(
                [
                    "ari",
                    "nmi",
                    "cluster_asw",
                    "calinski_harabasz",
                    "davies_bouldin",
                    "section_ari",
                    "section_nmi",
                    "section_asw",
                ]
            )
        )
        | (
            (summary["source_file"] == "spatial_continuity_summary.csv")
            & (summary["mode"] == "joint")
        )
        | (
            summary["source_file"].isin(
                ["clustering_metrics_by_label.csv", "external_clustering_metrics.csv"]
            )
            & (summary["mode"] == "joint")
            & (summary["scope"] == "combined")
            & summary["metric"].isin(["ari", "nmi"])
        )
    ].copy()

    lines = [
        "# result_v4C 与原版 result_v3 的逐 K 指标对比",
        "",
        "## 口径",
        "",
        "本报告比较相同数据集、相同分析模式、相同 scope、相同标签、相同聚类 K 和相同指标下的 result_v4C 与 result_v3。差值统一定义为 `v4C - v3`。ARI、NMI、ASW、CH、空间连续性等指标越高越好；DBI 越低越好；section ARI/NMI/ASW 越接近 0，表示 section mixing 越好。这里的 K 是聚类簇数，不是空间 KNN 的邻居数，也不是 OT top-k。",
        "",
        "完整逐 K 数值见 `V3_V4C_BY_K_LONG.csv`，逐指标统计见 `V3_V4C_BY_K_SUMMARY.csv`，横向展开的逐 K 差值见 `V3_V4C_BY_K_DELTA_WIDE.csv`。下面仅列主分析口径，完整 CSV 还包含 homogeneity、completeness、V-measure、label ASW 及 independent 模式等全部可对齐指标。",
        "",
        "## 比较覆盖",
        "",
        "| 数据集 | K 范围 | 逐 K 指标记录数 |",
        "|---|---:|---:|",
    ]
    for dataset in DATASETS.values():
        subset = long_df[long_df["dataset"] == dataset]
        k_values = sorted(subset["k"].unique())
        k_text = ", ".join(str(value) for value in k_values)
        lines.append(f"| {dataset} | {k_text} | {len(subset)} |")

    lines += ["", "## 主指标跨 K 汇总", ""]
    metric_names = {
        "ari": "ARI",
        "nmi": "NMI",
        "cluster_asw": "聚类 ASW",
        "calinski_harabasz": "CH",
        "davies_bouldin": "DBI",
        "section_ari": "section ARI",
        "section_nmi": "section NMI",
        "section_asw": "section ASW",
        "mean_spatial_neighbor_agreement": "空间邻居一致率",
        "neighbor_same_cluster_fraction": "空间邻居一致率",
    }
    for dataset in DATASETS.values():
        subset = primary[primary["dataset"] == dataset]
        lines += [f"### {dataset}", ""]
        if subset.empty:
            lines += ["没有可对齐的主指标。", ""]
            continue
        lines += [
            "| 指标/标签 | K | 提升/下降/持平 | 平均差值 | 最小差值（K） | 最大差值（K） |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in subset.sort_values(["source_file", "label", "metric"]).itertuples(index=False):
            label = ""
            if isinstance(row.label, str) and row.label:
                label = f" / {row.label}"
            metric = metric_names.get(row.metric, row.metric)
            lines.append(
                f"| {metric}{label} | {row.k_values} | "
                f"{row.improved_k_count}/{row.declined_k_count}/{row.equal_k_count} | "
                f"{format_number(row.mean_delta)} | "
                f"{format_number(row.min_delta)}（K={row.min_delta_k}） | "
                f"{format_number(row.max_delta)}（K={row.max_delta_k}） |"
            )
        lines.append("")

    lines += [
        "## 阅读注意",
        "",
        "- `提升/下降/持平` 已按指标方向判定，而差值列始终是原始的 `v4C-v3`；所以 DBI 的负差值通常代表提升。",
        "- section 指标按绝对值是否更接近 0 判定，不能只看差值正负。",
        "- label ASW 由 embedding 与标签直接计算，在同一 scope 内可能随 K 重复；完整表保留它，但不把这种重复值当作聚类 K 敏感性证据。",
        "- 本次只有 seed 42；逐 K 的优势或劣势是单次运行结果，不等同于多 seed 统计显著性。",
        "- CH 数值尺度远大于 0—1 指标，应看同一指标内部变化，不应与 ARI/NMI 的绝对差值直接比较。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = []
    for dataset in DATASETS:
        for filename in TABLES[dataset]:
            rows.extend(compare_table(dataset, filename))
    long_df = pd.DataFrame(rows)
    sort_cols = ["dataset", "source_file", "mode", "scope", "section", "label", "metric", "k"]
    long_df = long_df.sort_values(sort_cols, na_position="first").reset_index(drop=True)

    summary = make_summary(long_df).sort_values(
        ["dataset", "source_file", "mode", "scope", "section", "label", "metric"],
        na_position="first",
    )
    wide_index = [
        "dataset",
        "dataset_key",
        "source_file",
        "mode",
        "scope",
        "section",
        "label",
        "metric",
        "preferred_direction",
    ]
    wide = long_df.pivot(index=wide_index, columns="k", values="delta_v4C_minus_v3").reset_index()
    wide.columns = [f"K{column}" if isinstance(column, (int, np.integer)) else column for column in wide.columns]

    long_path = OUT_DIR / "V3_V4C_BY_K_LONG.csv"
    summary_path = OUT_DIR / "V3_V4C_BY_K_SUMMARY.csv"
    wide_path = OUT_DIR / "V3_V4C_BY_K_DELTA_WIDE.csv"
    report_path = OUT_DIR / "V3_V4C_BY_K_REPORT.md"
    long_df.to_csv(long_path, index=False, float_format="%.10g")
    summary.to_csv(summary_path, index=False, float_format="%.10g")
    wide.to_csv(wide_path, index=False, float_format="%.10g")
    write_report(long_df, summary, report_path)
    print(f"Wrote {len(long_df)} exact comparisons to {long_path}")
    print(f"Wrote {len(summary)} grouped summaries to {summary_path}")
    print(f"Wrote {len(wide)} wide delta rows to {wide_path}")
    print(f"Wrote Chinese report to {report_path}")


if __name__ == "__main__":
    main()
