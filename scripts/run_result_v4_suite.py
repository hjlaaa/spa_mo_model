#!/usr/bin/env python3
"""Run the six requested result_v4 trainings and standardized analyses serially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v4"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "suite_status.json"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def commands(python: str) -> list[tuple[str, list[str]]]:
    mousebrain_output = (
        RESULT_ROOT
        / "mousebrain"
        / "bidirectional_sparse_uot_fixed_lc0.1_seed42"
    )
    misar_output = (
        RESULT_ROOT
        / "misar_seq"
        / "bidirectional_sparse_uot_fixed_lc0.1_seed42"
    )
    return [
        (
            "train_mousebrain",
            [
                python,
                str(ROOT / "scripts/run_mousebrain_v2.py"),
                "--config", str(ROOT / "data/configs/mousebrain_preprocess_train.json"),
                "--epochs", "200",
                "--lambda_contrast", "0.1",
                "--device", "cuda",
                "--output_dir", str(mousebrain_output),
                "--seed", "42",
                "--ot_prior_mode", "candidate_sparse",
                "--bidirectional_ot_attention",
                "--candidate_backend", "faiss_ivf",
                "--initial_modality_candidate_k", "100",
                "--candidate_k", "200",
                "--attention_topk", "10",
                "--faiss_nlist", "256",
                "--faiss_nprobe", "32",
                "--faiss_device", "auto",
                "--faiss_train_sample_size", "10000",
                "--faiss_query_batch_size", "2048",
                "--dynamic_candidate_source", "fused",
                "--uot_epsilon", "0.05",
                "--uot_tau_a", "1.0",
                "--uot_tau_b", "1.0",
                "--uot_max_iter", "100",
                "--update_interval", "20",
                "--spatial_knn_k", "5",
                "--save_ot_prior_topk",
            ],
        ),
        ("train_human_lymph_node", [python, str(ROOT / "scripts/run_human_lymph_node.py")]),
        (
            "train_misar_seq",
            [
                python,
                str(ROOT / "scripts/run_misar_seq.py"),
                "--output_dir", str(misar_output),
                "--train",
                "--epochs", "200",
                "--seed", "42",
                "--device", "cuda",
                "--hvg_num", "3000",
                "--hvg_num_atac", "3000",
                "--lambda_contrast", "0.1",
                "--ot_prior_mode", "candidate_sparse",
                "--bidirectional_ot_attention",
                "--candidate_backend", "faiss_ivf",
                "--faiss_nlist", "128",
                "--faiss_nprobe", "32",
                "--faiss_train_sample_size", "20000",
                "--faiss_query_batch_size", "2048",
                "--initial_modality_candidate_k", "100",
                "--candidate_k", "200",
                "--attention_topk", "10",
                "--dynamic_candidate_source", "fused",
                "--spatial_knn_k", "5",
                "--graphsage_edge_batch_size", "50000",
                "--training_loss_only",
                "--decoder_chunk_size", "50000",
                "--ot_attention_source_chunk_size", "50000",
                "--checkpoint_ot_attention",
                "--checkpoint_encoder_fusion",
                "--checkpoint_decoder_chunks",
                "--checkpoint_graph_encoder",
                "--amp_dtype", "none",
                "--cache_spatial_graphs",
                "--save_candidate_qc",
                "--save_outputs",
                "--save_embeddings",
                "--save_ot_prior_topk",
                "--log_cuda_memory",
            ],
        ),
        ("train_mouse_spleen", [python, str(ROOT / "scripts/run_mouse_spleen.py")]),
        ("train_mouse_thymus", [python, str(ROOT / "scripts/run_mouse_thymus.py")]),
        ("train_simulation", [python, str(ROOT / "scripts/run_simulation.py")]),
        (
            "analyze_mousebrain",
            [python, str(ROOT / "scripts/analyze_mousebrain_v4_standardized.py")],
        ),
        (
            "analyze_other_five",
            [
                python,
                str(ROOT / "scripts/analyze_result_v3_standardized.py"),
                "--result-root", str(RESULT_ROOT),
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--from-task", default=None)
    args = parser.parse_args()

    task_list = commands(args.python)
    task_names = [name for name, _ in task_list]
    if args.from_task:
        if args.from_task not in task_names:
            raise ValueError(f"unknown --from-task {args.from_task!r}: {task_names}")
        task_list = task_list[task_names.index(args.from_task):]

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status = {
        "suite": "result_v4_microenvironment_aware",
        "started_at": now(),
        "python": args.python,
        "dataset_order": [
            "dataset_MouseBrain",
            "Human_Lymph_Node",
            "MISAR-seq",
            "Mouse_Spleen",
            "Mouse_Thymus",
            "Simulation",
        ],
        "seed": 42,
        "epochs": 200,
        "lambda_contrast": 0.1,
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "dynamic_candidate_source": "fused",
        "tasks": {},
    }
    write_json(STATUS_PATH, status)

    for name, command in task_list:
        log_path = LOG_ROOT / f"{name}.log"
        task_status = {
            "status": "running",
            "started_at": now(),
            "command": command,
            "log": str(log_path),
        }
        status["tasks"][name] = task_status
        write_json(STATUS_PATH, status)
        print(f"[{now()}] START {name}: {' '.join(command)}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        task_status.update(
            {
                "status": "completed" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "finished_at": now(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(STATUS_PATH, status)
        print(
            f"[{now()}] {task_status['status'].upper()} {name} "
            f"rc={process.returncode} log={log_path}",
            flush=True,
        )
        if process.returncode != 0:
            status["status"] = "failed"
            status["failed_task"] = name
            status["finished_at"] = now()
            write_json(STATUS_PATH, status)
            raise SystemExit(process.returncode)

    status["status"] = "completed"
    status["finished_at"] = now()
    write_json(STATUS_PATH, status)
    print(f"[{now()}] RESULT_V4_SUITE: PASS", flush=True)


if __name__ == "__main__":
    main()
