#!/usr/bin/env python3
"""Detached serial runner for result_v4 CRC Stereo-CITE-seq and SPATCH."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v4"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "large_suite_status.json"
PID_PATH = RESULT_ROOT / "large_suite.pid"
SUPERVISOR_LOG = LOG_ROOT / "large_suite_supervisor.log"
VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def task_commands(python: str) -> list[tuple[str, list[str]]]:
    crc_output = RESULT_ROOT / "crc_stereocite" / VARIANT
    spatch_output = RESULT_ROOT / "spatch" / VARIANT
    return [
        (
            "train_crc_stereocite",
            [
                python,
                str(ROOT / "scripts/run_crc_stereocite.py"),
                "--data_dir", str(ROOT / "data/CRC_Stereo-CITE-seq"),
                "--output_dir", str(crc_output),
                "--train",
                "--epochs", "200",
                "--seed", "42",
                "--device", "cuda",
                "--max_shared_genes", "3000",
                "--hvg_num", "3000",
                "--lambda_contrast", "0.1",
                "--ot_prior_mode", "candidate_sparse",
                "--bidirectional_ot_attention",
                "--candidate_backend", "faiss_ivf",
                "--faiss_nlist", "4096",
                "--faiss_nprobe", "64",
                "--faiss_device", "auto",
                "--faiss_train_sample_size", "150000",
                "--faiss_query_batch_size", "2048",
                "--initial_modality_candidate_k", "100",
                "--candidate_k", "200",
                "--attention_topk", "10",
                "--dynamic_candidate_source", "fused",
                "--enable_context_attention_gate",
                "--uot_epsilon", "0.05",
                "--uot_tau_a", "1.0",
                "--uot_tau_b", "1.0",
                "--uot_max_iter", "100",
                "--update_interval", "20",
                "--spatial_knn_k", "5",
                "--graphsage_edge_batch_size", "100000",
                "--training_loss_only",
                "--decoder_chunk_size", "50000",
                "--ot_attention_source_chunk_size", "50000",
                "--checkpoint_ot_attention",
                "--checkpoint_encoder_fusion",
                "--checkpoint_decoder_chunks",
                "--checkpoint_graph_encoder",
                "--amp_dtype", "bf16",
                "--cache_spatial_graphs",
                "--save_candidate_qc",
                "--save_outputs",
                "--save_embeddings",
                "--save_ot_prior_topk",
                "--log_cuda_memory",
                "--log_cuda_memory_detail",
            ],
        ),
        (
            "train_spatch",
            [
                python,
                str(ROOT / "scripts/run_spatch.py"),
                "--data_dir", str(ROOT / "data/spatch"),
                "--output_dir", str(spatch_output),
                "--epochs", "200",
                "--seed", "42",
                "--device", "cuda",
                "--n_comps", "50",
                "--hvg_num", "3000",
                "--lambda_contrast", "0.1",
                "--update_interval", "20",
                "--uot_max_iter", "100",
                "--candidate_backend", "faiss_ivf",
                "--initial_modality_candidate_k", "100",
                "--candidate_k", "200",
                "--attention_topk", "10",
                "--faiss_nlist", "4096",
                "--faiss_nprobe", "64",
                "--faiss_device", "gpu",
                "--faiss_train_sample_size", "100000",
                "--faiss_query_batch_size", "2048",
                "--dynamic_candidate_source", "fused",
                "--enable_context_attention_gate",
                "--uot_epsilon", "0.05",
                "--uot_tau_a", "1.0",
                "--uot_tau_b", "1.0",
                "--spatial_knn_k", "10",
                "--graphsage_edge_batch_size", "100000",
                "--decoder_chunk_size", "8192",
                "--ot_attention_source_chunk_size", "2048",
                "--amp_dtype", "bf16",
            ],
        ),
        (
            "analyze_large_datasets",
            [
                python,
                str(ROOT / "scripts/analyze_result_v3_large_standardized.py"),
                "--result-root", str(RESULT_ROOT),
            ],
        ),
    ]


def launch_daemon(python: str) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [python, str(Path(__file__).resolve()), "--python", python]
    with SUPERVISOR_LOG.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pid": process.pid,
                "pid_file": str(PID_PATH),
                "status_file": str(STATUS_PATH),
                "supervisor_log": str(SUPERVISOR_LOG),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_suite(python: str) -> None:
    tasks = task_commands(python)
    status = {
        "suite": "result_v4_large_microenvironment_aware",
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "dataset_order": ["CRC_Stereo-CITE-seq", "spatch"],
        "epochs": 200,
        "seed": 42,
        "lambda_contrast": 0.1,
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "dynamic_candidate_source": "fused",
        "tasks": {},
        "status": "running",
    }
    write_json(STATUS_PATH, status)
    for name, command in tasks:
        log_path = LOG_ROOT / f"{name}.log"
        task = {
            "status": "running",
            "started_at": now(),
            "command": command,
            "log": str(log_path),
        }
        status["tasks"][name] = task
        write_json(STATUS_PATH, status)
        print(f"[{now()}] START {name}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        task.update(
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "finished_at": now(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(STATUS_PATH, status)
        print(f"[{now()}] {task['status'].upper()} {name}", flush=True)
        if result.returncode != 0:
            status.update(
                {
                    "status": "failed",
                    "failed_task": name,
                    "finished_at": now(),
                }
            )
            write_json(STATUS_PATH, status)
            raise SystemExit(result.returncode)
    status.update({"status": "completed", "finished_at": now()})
    write_json(STATUS_PATH, status)
    print(f"[{now()}] RESULT_V4_LARGE_SUITE: PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if args.daemon:
        launch_daemon(args.python)
        return
    run_suite(args.python)


if __name__ == "__main__":
    main()
