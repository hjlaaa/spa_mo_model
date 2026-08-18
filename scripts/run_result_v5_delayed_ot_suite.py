#!/usr/bin/env python3
"""Run the three result_v5 delayed-OT experiments and their v3 analyses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v5"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "suite_status.json"
MOUSE_RUN = "v3_bidirectional_sparse_fixed_lc0.1"
OTHER_RUN = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
EXPECTED_OT_UPDATES = [100, 120, 140, 160, 180, 200]


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def gpu_check(python: str, label: str, attempts: int = 5) -> dict:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        print(f"[{now()}] GPU check {label} attempt {attempt}/{attempts}", flush=True)
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        probe = subprocess.run(
            [
                python,
                "-c",
                (
                    "import torch; assert torch.cuda.is_available(); "
                    "x=torch.ones((512,512),device='cuda'); y=x@x; "
                    "torch.cuda.synchronize(); "
                    "print(torch.cuda.get_device_name(0), float(y[0,0]))"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0 and probe.returncode == 0:
            detail = {
                "checked_at": now(),
                "attempt": attempt,
                "nvidia_smi": smi.stdout.strip(),
                "torch_probe": probe.stdout.strip(),
            }
            print(f"[{now()}] GPU check {label}: PASS", flush=True)
            return detail
        errors.append(
            f"attempt={attempt} nvidia_smi={smi.stderr.strip()} torch={probe.stderr.strip()}"
        )
        if attempt < attempts:
            time.sleep(5)
    raise RuntimeError(f"GPU unavailable after {attempts} checks: {errors}")


def commands(python: str) -> list[tuple[str, list[str], bool]]:
    mouse_output = RESULT_ROOT / "mousebrain" / MOUSE_RUN
    misar_output = RESULT_ROOT / "misar_seq" / OTHER_RUN
    embryo_output = RESULT_ROOT / "human_embryo_harmony"
    embryo_preprocessed = Path("/home/hujinlan/human_embryo/human_embryo_harmony")
    hierarchy_k = "2,3,4,10,15,20,25"
    return [
        (
            "train_mousebrain",
            [
                python, str(ROOT / "scripts/run_mousebrain_v2.py"),
                "--config", str(ROOT / "data/configs/mousebrain_preprocess_train.json"),
                "--epochs", "200", "--lambda_contrast", "0.1",
                "--device", "cuda", "--output_dir", str(mouse_output),
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
            True,
        ),
        (
            "train_human_embryo",
            [
                python, str(ROOT / "scripts/run_human_embryo_rna_only.py"),
                "--data_dir", "/home/hujinlan/human_embryo",
                "--output_dir", str(embryo_output), "--train",
                "--single_modality", "RNA", "--reuse_preprocessed",
                "--preprocessed_run_dir", str(embryo_preprocessed),
                "--require_harmony", "--epochs", "200", "--lr", "0.001",
                "--weight_decay", "0", "--device", "cuda", "--seed", "42",
                "--log_every", "1", "--update_interval", "20",
                "--uot_max_iter", "100", "--uot_epsilon", "0.05",
                "--uot_tau_a", "1.0", "--uot_tau_b", "1.0",
                "--bidirectional_ot_attention", "--candidate_backend", "faiss_ivf",
                "--initial_modality_candidate_k", "100", "--candidate_k", "200",
                "--attention_topk", "10", "--faiss_nlist", "4096",
                "--faiss_nprobe", "64", "--faiss_device", "auto",
                "--faiss_train_sample_size", "100000",
                "--faiss_query_batch_size", "2048",
                "--dynamic_candidate_source", "final", "--spatial_knn_k", "5",
                "--graphsage_edge_batch_size", "200000",
                "--decoder_chunk_size", "50000",
                "--ot_attention_source_chunk_size", "50000",
                "--training_loss_only", "--checkpoint_ot_attention",
                "--checkpoint_encoder_fusion", "--checkpoint_decoder_chunks",
                "--checkpoint_graph_encoder", "--cache_spatial_graphs",
                "--amp_dtype", "bf16", "--log_cuda_memory",
            ],
            True,
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
                "--faiss_nprobe", "32", "--faiss_device", "cpu",
                "--faiss_train_sample_size", "20000",
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
                "--save_ot_prior_topk", "--update_interval", "20",
            ],
            True,
        ),
        (
            "analyze_mousebrain",
            [
                python, str(ROOT / "scripts/analyze_mousebrain_v4_standardized.py"),
                "--result-root", str(RESULT_ROOT), "--run-name", MOUSE_RUN,
            ],
            False,
        ),
        (
            "analyze_human_embryo",
            [
                python, str(ROOT / "scripts/analyze_human_embryo_reference_metrics.py"),
                "--input-dir", str(embryo_output),
                "--output-dir", str(embryo_output / "analysis"),
                "--joint-k", hierarchy_k, "--independent-k", hierarchy_k,
                "--plot-k", hierarchy_k, "--reuse-cache", "--reuse-flat-labels",
            ],
            False,
        ),
        (
            "analyze_misar_seq",
            [
                python, str(ROOT / "scripts/analyze_misar_result_v5_standardized.py"),
                "--result-root", str(RESULT_ROOT),
            ],
            False,
        ),
    ]


def validate_training_summary(task_name: str) -> None:
    paths = {
        "train_mousebrain": RESULT_ROOT / "mousebrain" / MOUSE_RUN / "epochs_200" / "run_summary.json",
        "train_human_embryo": RESULT_ROOT / "human_embryo_harmony" / "run_summary.json",
        "train_misar_seq": RESULT_ROOT / "misar_seq" / OTHER_RUN / "run_summary.json",
    }
    summary = json.loads(paths[task_name].read_text())
    if summary.get("ot_updates") != EXPECTED_OT_UPDATES:
        raise ValueError(
            f"{task_name}: unexpected OT schedule {summary.get('ot_updates')}"
        )
    if task_name == "train_human_embryo":
        if summary.get("status") != "PASS" or summary.get("gpu_name") is None:
            raise ValueError("Human embryo summary does not confirm successful GPU training.")
    elif task_name == "train_misar_seq":
        if summary.get("mode") != "train" or summary.get("epochs") != 200:
            raise ValueError("MISAR-seq summary does not confirm 200-epoch training.")
    else:
        if summary.get("mode") != "train" or summary.get("epochs") != 200:
            raise ValueError("MouseBrain summary does not confirm 200-epoch training.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-task",
        default=None,
        help="Resume at a named task while preserving the existing suite status.",
    )
    args = parser.parse_args()
    python = sys.executable
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    new_status = {
        "suite": "result_v5_delayed_ot_refresh",
        "started_at": now(),
        "python": python,
        "datasets": ["MouseBrain", "Human Embryo", "MISAR-seq"],
        "excluded_datasets": ["SPATCH"],
        "baseline": "result_v3",
        "mousebrain_baseline": str(
            ROOT / "result_v3/mousebrain/v3_bidirectional_sparse_fixed_lc0.1/epochs_200"
        ),
        "expected_ot_updates": EXPECTED_OT_UPDATES,
        "status": "running",
        "tasks": {},
    }
    status = (
        json.loads(STATUS_PATH.read_text())
        if args.from_task and STATUS_PATH.is_file()
        else new_status
    )
    status["status"] = "running"
    status.pop("failed_task", None)
    status.pop("finished_at", None)
    write_json(STATUS_PATH, status)
    task_list = commands(python)
    task_names = [name for name, _, _ in task_list]
    if args.from_task:
        if args.from_task not in task_names:
            raise ValueError(f"Unknown --from-task {args.from_task!r}: {task_names}")
        task_list = task_list[task_names.index(args.from_task):]
    for name, command, needs_gpu in task_list:
        task = {
            "status": "checking_gpu" if needs_gpu else "running",
            "started_at": now(),
            "command": command,
            "log": str(LOG_ROOT / f"{name}.log"),
        }
        status["tasks"][name] = task
        write_json(STATUS_PATH, status)
        if needs_gpu:
            task["gpu_check"] = gpu_check(python, name)
            task["status"] = "running"
            write_json(STATUS_PATH, status)
        print(f"[{now()}] START {name}", flush=True)
        started = time.perf_counter()
        with (LOG_ROOT / f"{name}.log").open("w", encoding="utf-8") as log:
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
        if result.returncode == 0 and name.startswith("train_"):
            validate_training_summary(name)
            task["ot_schedule_validated"] = True
        write_json(STATUS_PATH, status)
        if result.returncode != 0:
            status.update({"status": "failed", "failed_task": name, "finished_at": now()})
            write_json(STATUS_PATH, status)
            raise SystemExit(result.returncode)
        print(f"[{now()}] COMPLETED {name}", flush=True)
    status.update({"status": "completed", "finished_at": now()})
    write_json(STATUS_PATH, status)
    print(f"[{now()}] RESULT_V5_DELAYED_OT_SUITE: PASS", flush=True)


if __name__ == "__main__":
    main()
