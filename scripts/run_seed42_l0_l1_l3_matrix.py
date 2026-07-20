#!/usr/bin/env python3
"""Run the fixed seed-42 L0/L1/L3 contrastive experiment matrix.

The matrix is intentionally narrow:

* L1 corrected COSIE on MouseBrain, MISAR-seq, Human Lymph Node,
  Mouse Spleen, Mouse Thymus, and CRC Stereo-CITE-seq.
* L0 legacy and L3 InfoNCE+VICReg are additionally trained only on CRC,
  where the v2 result tree did not previously contain either baseline.
* Every model training seed is 42.  Every completed model is evaluated with
  clustering seeds 0..4, while metric subsampling seeds remain fixed at 0.

The runner is resumable.  A complete 200-epoch training history or completed
analysis manifest is skipped, and each command writes a dedicated log below
``logs/seed42_l0_l1_l3_matrix``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "result_v2_Contrastive_loss"
LOG_DIR = ROOT / "logs" / "seed42_l0_l1_l3_matrix"
PYTHON = Path(sys.executable)
MODEL_SEED = 42
CLUSTER_SEEDS = range(5)


@dataclass(frozen=True)
class TrainingSpec:
    dataset: str
    experiment: str
    run_dir: Path
    command: tuple[str, ...]


def _cmd(script: str, *args: str) -> tuple[str, ...]:
    return (str(PYTHON), str(ROOT / "scripts" / script), *map(str, args))


def _l1_dir(dataset: str, mousebrain: bool = False) -> Path:
    base = RESULTS / dataset / "L1_corrected_cosie_strict_init_matched" / "seed_42"
    return base / "epochs_200" if mousebrain else base


def training_specs() -> list[TrainingSpec]:
    mousebrain_parent = (
        RESULTS / "mousebrain" / "L1_corrected_cosie_strict_init_matched" / "seed_42"
    )
    mousebrain_run = mousebrain_parent / "epochs_200"
    mousebrain = TrainingSpec(
        "mousebrain",
        "L1_corrected_cosie_strict_init_matched",
        mousebrain_run,
        _cmd(
            "run_mousebrain_v2.py",
            "--config", "data/configs/mousebrain_preprocess_train.json",
            "--epochs", "200",
            "--seed", "42",
            "--device", "cuda",
            "--output_dir", str(mousebrain_parent.relative_to(ROOT)),
            "--contrastive_method", "corrected_cosie_dimension",
            "--lambda_contrast_schedule",
            "1-5:1e-4,6-10:3e-4,11-15:1e-3,16-200:1e-2",
        ),
    )

    misar_run = _l1_dir("misar_seq")
    misar = TrainingSpec(
        "misar_seq",
        "L1_corrected_cosie_strict_init_matched",
        misar_run,
        _cmd(
            "run_misar_seq.py",
            "--train", "--epochs", "200", "--seed", "42", "--device", "cuda",
            "--output_dir", str(misar_run.relative_to(ROOT)),
            "--contrastive_method", "corrected_cosie_dimension",
            "--lambda_contrast", "0.1",
            "--ot_prior_mode", "candidate_sparse", "--bidirectional_ot_attention",
            "--candidate_backend", "faiss_ivf", "--faiss_nlist", "128",
            "--faiss_nprobe", "32", "--faiss_device", "cpu",
            "--faiss_train_sample_size", "20000", "--faiss_query_batch_size", "2048",
            "--initial_modality_candidate_k", "100", "--candidate_k", "200",
            "--attention_topk", "10", "--spatial_knn_k", "5",
            "--graphsage_edge_batch_size", "50000", "--training_loss_only",
            "--decoder_chunk_size", "50000", "--ot_attention_source_chunk_size", "50000",
            "--checkpoint_ot_attention", "--checkpoint_encoder_fusion",
            "--checkpoint_decoder_chunks", "--checkpoint_graph_encoder",
            "--amp_dtype", "none", "--cache_spatial_graphs", "--save_candidate_qc",
            "--save_outputs", "--save_embeddings", "--save_ot_prior_topk",
            "--log_cuda_memory",
        ),
    )

    wrapper_specs = []
    for dataset, script in (
        ("human_lymph_node", "run_human_lymph_node.py"),
        ("mouse_spleen", "run_mouse_spleen.py"),
        ("mouse_thymus", "run_mouse_thymus.py"),
    ):
        run_dir = _l1_dir(dataset)
        wrapper_specs.append(
            TrainingSpec(
                dataset,
                "L1_corrected_cosie_strict_init_matched",
                run_dir,
                _cmd(
                    script,
                    "--output_dir", str(run_dir.relative_to(ROOT)),
                    "--contrastive_method", "corrected_cosie_dimension",
                    "--device", "cuda", "--amp_dtype", "none",
                    "--faiss_device", "cpu",
                ),
            )
        )

    crc_common = (
        "--train", "--epochs", "200", "--seed", "42", "--device", "cuda",
        "--max_shared_genes", "10000", "--hvg_num", "3000",
        "--lambda_contrast", "0.1", "--ot_prior_mode", "candidate_sparse",
        "--bidirectional_ot_attention", "--candidate_backend", "faiss_ivf",
        "--faiss_nlist", "4096", "--faiss_nprobe", "64", "--faiss_device", "auto",
        "--faiss_train_sample_size", "150000", "--faiss_query_batch_size", "2048",
        "--initial_modality_candidate_k", "100", "--candidate_k", "200",
        "--attention_topk", "10", "--spatial_knn_k", "10",
        "--graphsage_edge_batch_size", "100000", "--training_loss_only",
        "--decoder_chunk_size", "50000", "--ot_attention_source_chunk_size", "50000",
        "--checkpoint_ot_attention", "--checkpoint_encoder_fusion",
        "--checkpoint_decoder_chunks", "--checkpoint_graph_encoder",
        "--amp_dtype", "bf16", "--cache_spatial_graphs", "--save_candidate_qc",
        "--save_outputs", "--save_embeddings", "--save_ot_prior_topk",
        "--log_cuda_memory",
    )
    crc_specs = []
    for experiment, method in (
        ("L0_legacy_strict_init_matched", "legacy_cosie_dimension"),
        ("L1_corrected_cosie_strict_init_matched", "corrected_cosie_dimension"),
        ("L3_vicreg_strict_init_matched", "symmetric_infonce_vicreg"),
    ):
        run_dir = RESULTS / "crc_stereocite" / experiment / "seed_42"
        crc_specs.append(
            TrainingSpec(
                "crc_stereocite",
                experiment,
                run_dir,
                _cmd(
                    "run_crc_stereocite.py",
                    *crc_common,
                    "--output_dir", str(run_dir.relative_to(ROOT)),
                    "--contrastive_method", method,
                ),
            )
        )
    return [mousebrain, misar, *wrapper_specs, *crc_specs]


def _training_complete(spec: TrainingSpec) -> bool:
    summary_path = spec.run_dir / "run_summary.json"
    history_path = spec.run_dir / "loss_history.json"
    if not summary_path.exists() or not history_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    epochs = int(summary.get("epochs", 0))
    if epochs == 0:
        epochs = int(summary.get("preprocessing", {}).get("resolved_model_config", {}).get("training", {}).get("epochs", 0))
    return epochs == 200 and len(history) == 200


def _run(name: str, command: tuple[str, ...]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    print(f"START {name}", flush=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}; see {log_path}"
        )
    print(f"DONE {name} elapsed={time.time() - started:.1f}s", flush=True)


def train(specs: list[TrainingSpec]) -> None:
    for spec in specs:
        name = f"train_{spec.dataset}_{spec.experiment}_seed42"
        if _training_complete(spec):
            print(f"SKIP {name}: complete", flush=True)
            continue
        _run(name, spec.command)
        if not _training_complete(spec):
            raise RuntimeError(f"{name} exited successfully but completion validation failed")


def _base_analysis_command(spec: TrainingSpec, seed: int, analysis_dir: Path) -> tuple[str, ...]:
    relative_run = str(spec.run_dir.relative_to(ROOT))
    relative_analysis = str(analysis_dir.relative_to(ROOT))
    if spec.dataset == "mousebrain":
        return _cmd(
            "analyze_mousebrain_clustering.py",
            "--embedding_dir", str((spec.run_dir / "final_embeddings").relative_to(ROOT)),
            "--config", "data/configs/mousebrain_preprocess_train.json",
            "--output_dir", str((analysis_dir / "clustering").relative_to(ROOT)),
            "--cluster_list", "5,6,8,9,10,11", "--random_state", str(seed),
            "--metric_seed", "0", "--batch_metric_seed", "0",
            "--label_keys", "annotations,RegionLoupe",
        )
    if spec.dataset == "misar_seq":
        return _cmd(
            "analyze_misar_seq_clustering.py",
            "--input_dir", relative_run, "--output_dir", relative_analysis,
            "--section_order", "dataset4,dataset3,dataset2,dataset1",
            "--n_clusters", "8,10,12,14",
            "--label_keys", "Y,Combined_Clusters_annotation,Combined_Clusters,RNA_Clusters,ATAC_Clusters",
            "--seed", str(seed), "--metric_seed", "0", "--batch_metric_seed", "0",
            "--kmeans_method", "kmeans", "--n_init", "20",
            "--metric_sample_size", "10000", "--batch_metrics_max_samples", "0",
        )
    if spec.dataset == "human_lymph_node":
        return _cmd(
            "analyze_human_lymph_node.py", "--input_dir", relative_run,
            "--output_dir", relative_analysis, "--seed", str(seed),
            "--metric_seed", "0", "--batch_metric_seed", "0",
        )
    if spec.dataset == "mouse_spleen":
        return _cmd(
            "analyze_mouse_spleen.py", "--input_dir", relative_run,
            "--output_dir", relative_analysis, "--seed", str(seed),
            "--metric_seed", "0", "--batch_metric_seed", "0",
        )
    if spec.dataset == "mouse_thymus":
        return _cmd(
            "analyze_mouse_thymus.py", "--input_dir", relative_run,
            "--output_dir", relative_analysis, "--seed", str(seed),
            "--metric_seed", "0", "--batch_metric_seed", "0",
        )
    if spec.dataset == "crc_stereocite":
        return _cmd(
            "analyze_crc_stereocite_clustering.py",
            "--input_dir", relative_run, "--output_dir", relative_analysis,
            "--n_clusters", "5,8,10,15", "--seed", str(seed),
            "--metric_seed", "0", "--batch_metric_seed", "0",
            "--kmeans_method", "minibatch", "--batch_size", "8192",
            "--n_init", "10", "--plot_max_points", "50000",
            "--batch_metrics_max_samples", "50000",
        )
    raise ValueError(spec.dataset)


def _completion_command(spec: TrainingSpec, analysis_dir: Path) -> tuple[str, ...]:
    clustering_dir = analysis_dir / "clustering" if spec.dataset == "mousebrain" else analysis_dir
    dataset_names = {
        "mousebrain": "MouseBrain",
        "misar_seq": "MISAR-seq",
        "human_lymph_node": "Human Lymph Node",
        "mouse_spleen": "Mouse Spleen",
        "mouse_thymus": "Mouse Thymus",
        "crc_stereocite": "CRC Stereo-CITE-seq",
    }
    sections = {
        "mousebrain": ("s1", "s2", "s3"),
        "misar_seq": ("dataset4", "dataset3", "dataset2", "dataset1"),
        "human_lymph_node": ("Human_Lymph_Node_A1", "Human_Lymph_Node_D1"),
        "mouse_spleen": ("Mouse_Spleen1", "Mouse_Spleen2"),
        "mouse_thymus": tuple(f"Mouse_Thymus{i}" for i in range(1, 5)),
        "crc_stereocite": ("CRC_003", "CRC_006"),
    }[spec.dataset]
    metric_sample_size = "5000" if spec.dataset == "mousebrain" else "10000"
    return _cmd(
        "complete_spa_mo_analysis.py",
        "--dataset", dataset_names[spec.dataset],
        "--run-dir", str(spec.run_dir.relative_to(ROOT)),
        "--analysis-dir", str(analysis_dir.relative_to(ROOT)),
        "--clustering-dir", str(clustering_dir.relative_to(ROOT)),
        "--sections", *sections, "--seed", "0",
        "--metric-sample-size", metric_sample_size,
        "--spatial-neighbor-k", "6",
    )


def evaluate(specs: list[TrainingSpec]) -> None:
    for spec in specs:
        if not _training_complete(spec):
            raise RuntimeError(f"Cannot evaluate incomplete training: {spec.run_dir}")
        for seed in CLUSTER_SEEDS:
            analysis_dir = spec.run_dir / "evaluation" / f"cluster_seed_{seed}"
            manifest = analysis_dir / "analysis_completion_manifest.json"
            name = f"eval_{spec.dataset}_{spec.experiment}_model42_cluster{seed}"
            if manifest.exists():
                print(f"SKIP {name}: complete", flush=True)
                continue
            _run(name + "_cluster", _base_analysis_command(spec, seed, analysis_dir))
            _run(name + "_metrics", _completion_command(spec, analysis_dir))
            if not manifest.exists():
                raise RuntimeError(f"{name}: completion manifest was not created")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("all", "train", "evaluate"), default="all"
    )
    parser.add_argument(
        "--scope", choices=("all", "l1", "crc"), default="all",
        help="l1 selects six L1 runs; crc selects CRC L0/L1/L3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = training_specs()
    if args.scope == "l1":
        specs = [spec for spec in specs if spec.experiment.startswith("L1_")]
    elif args.scope == "crc":
        specs = [spec for spec in specs if spec.dataset == "crc_stereocite"]
    if args.phase in {"all", "train"}:
        train(specs)
    if args.phase in {"all", "evaluate"}:
        evaluate(specs)
    print("SEED42_L0_L1_L3_MATRIX: PASS", flush=True)


if __name__ == "__main__":
    main()
