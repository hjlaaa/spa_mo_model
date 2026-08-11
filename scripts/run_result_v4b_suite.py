#!/usr/bin/env python3
"""Run experiment B: v3 dynamic OT plus microenvironment attention gate."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v4B"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "suite_status.json"
VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def commands(python: str) -> list[tuple[str, list[str]]]:
    mousebrain_output = RESULT_ROOT / "mousebrain" / VARIANT
    hln_output = RESULT_ROOT / "human_lymph_node" / VARIANT
    misar_output = RESULT_ROOT / "misar_seq" / VARIANT
    spleen_output = RESULT_ROOT / "mouse_spleen" / VARIANT
    thymus_output = RESULT_ROOT / "mouse_thymus" / VARIANT
    simulation_output = RESULT_ROOT / "simulation" / VARIANT
    return [
        (
            "train_mousebrain",
            [
                python, str(ROOT / "scripts/run_mousebrain_v2.py"),
                "--config", str(ROOT / "data/configs/mousebrain_preprocess_train.json"),
                "--epochs", "200", "--lambda_contrast", "0.1",
                "--device", "cuda", "--output_dir", str(mousebrain_output),
                "--seed", "42", "--ot_prior_mode", "candidate_sparse",
                "--bidirectional_ot_attention", "--candidate_backend", "faiss_ivf",
                "--initial_modality_candidate_k", "100", "--candidate_k", "200",
                "--attention_topk", "10", "--faiss_nlist", "256",
                "--faiss_nprobe", "32", "--faiss_device", "auto",
                "--faiss_train_sample_size", "10000",
                "--faiss_query_batch_size", "2048",
                "--dynamic_candidate_source", "final",
                "--uot_epsilon", "0.05", "--uot_tau_a", "1.0",
                "--uot_tau_b", "1.0", "--uot_max_iter", "100",
                "--update_interval", "20", "--spatial_knn_k", "5",
                "--save_ot_prior_topk",
            ],
        ),
        (
            "train_human_lymph_node",
            [
                python, str(ROOT / "scripts/run_human_lymph_node.py"),
                "--output_dir", str(hln_output),
                "--dynamic_candidate_source", "final",
            ],
        ),
        (
            "train_misar_seq",
            [
                python, str(ROOT / "scripts/run_misar_seq.py"),
                "--output_dir", str(misar_output), "--train", "--epochs", "200",
                "--seed", "42", "--device", "cuda", "--hvg_num", "3000",
                "--hvg_num_atac", "3000", "--lambda_contrast", "0.1",
                "--ot_prior_mode", "candidate_sparse", "--bidirectional_ot_attention",
                "--candidate_backend", "faiss_ivf", "--faiss_nlist", "128",
                "--faiss_nprobe", "32", "--faiss_train_sample_size", "20000",
                "--faiss_query_batch_size", "2048",
                "--initial_modality_candidate_k", "100", "--candidate_k", "200",
                "--attention_topk", "10", "--dynamic_candidate_source", "final",
                "--spatial_knn_k", "5", "--graphsage_edge_batch_size", "50000",
                "--training_loss_only", "--decoder_chunk_size", "50000",
                "--ot_attention_source_chunk_size", "50000",
                "--checkpoint_ot_attention", "--checkpoint_encoder_fusion",
                "--checkpoint_decoder_chunks", "--checkpoint_graph_encoder",
                "--amp_dtype", "none", "--cache_spatial_graphs",
                "--save_candidate_qc", "--save_outputs", "--save_embeddings",
                "--save_ot_prior_topk", "--log_cuda_memory",
            ],
        ),
        (
            "train_mouse_spleen",
            [
                python, str(ROOT / "scripts/run_mouse_spleen.py"),
                "--output_dir", str(spleen_output),
                "--dynamic_candidate_source", "final",
            ],
        ),
        (
            "train_mouse_thymus",
            [
                python, str(ROOT / "scripts/run_mouse_thymus.py"),
                "--output_dir", str(thymus_output),
                "--dynamic_candidate_source", "final",
            ],
        ),
        (
            "train_simulation",
            [
                python, str(ROOT / "scripts/run_simulation.py"),
                "--output_dir", str(simulation_output),
                "--dynamic_candidate_source", "final",
            ],
        ),
        (
            "analyze_mousebrain",
            [
                python, str(ROOT / "scripts/analyze_mousebrain_v4_standardized.py"),
                "--result-root", str(RESULT_ROOT),
            ],
        ),
        (
            "analyze_other_five",
            [
                python, str(ROOT / "scripts/analyze_result_v3_standardized.py"),
                "--result-root", str(RESULT_ROOT),
            ],
        ),
    ]


def main() -> None:
    python = sys.executable
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status = {
        "suite": "result_v4B_v3_ot_plus_microenvironment_attention_gate",
        "started_at": now(),
        "dataset_order": [
            "dataset_MouseBrain", "Human_Lymph_Node", "MISAR-seq",
            "Mouse_Spleen", "Mouse_Thymus", "Simulation",
        ],
        "epochs": 200,
        "seed": 42,
        "lambda_contrast": 0.1,
        "ot_prior_mode": "candidate_sparse",
        "bidirectional_ot_attention": True,
        "dynamic_candidate_source": "final",
        "dynamic_semantic_source": "final",
        "dynamic_context_source": "final_spatial_context",
        "attention_context_source": "fused_spatial_context",
        "attention_context_gate_enabled": True,
        "status": "running",
        "tasks": {},
    }
    write_json(STATUS_PATH, status)
    for name, command in commands(python):
        log_path = LOG_ROOT / f"{name}.log"
        task = {
            "status": "running", "started_at": now(),
            "command": command, "log": str(log_path),
        }
        status["tasks"][name] = task
        write_json(STATUS_PATH, status)
        print(f"[{now()}] START {name}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
            )
        task.update(
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "finished_at": now(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(STATUS_PATH, status)
        print(f"[{now()}] {task['status'].upper()} {name}", flush=True)
        if result.returncode != 0:
            status.update(
                {"status": "failed", "failed_task": name, "finished_at": now()}
            )
            write_json(STATUS_PATH, status)
            raise SystemExit(result.returncode)
    status.update({"status": "completed", "finished_at": now()})
    write_json(STATUS_PATH, status)
    print(f"[{now()}] RESULT_V4B_SUITE: PASS", flush=True)


if __name__ == "__main__":
    main()
