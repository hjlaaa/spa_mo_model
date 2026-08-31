#!/usr/bin/env python3
"""Complete the five missing result_v5 datasets with v3-matched settings.

Training is executed from a clean checkout of commit a73e289 (the v5 delayed
OT-refresh model), so the uncommitted v6 model changes in the main worktree are
left untouched. Outputs and analyses are written to the main result_v5 tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/home/hujinlan/spa_mo_model")
V5_CODE_ROOT = Path("/home/hujinlan/spa_mo_model_v5_code")
V5_COMMIT = "a73e289521a4874ccc5aa3220a6858c7989a2b41"
RESULT_ROOT = ROOT / "result_v5"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "missing_v3_matched_suite_status.json"
PID_PATH = RESULT_ROOT / "missing_v3_matched_suite.pid"
EXIT_PATH = RESULT_ROOT / "missing_v3_matched_suite.exit"
SUPERVISOR_LOG = LOG_ROOT / "missing_v3_matched_suite_supervisor.log"
VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
EXPECTED_OT_UPDATES = [100, 120, 140, 160, 180, 200]
EXPECTED_RETAINED_K = {
    "human_lymph_node": [5, 8, 10, 12],
    "mouse_spleen": [5, 8, 10, 12],
    "mouse_thymus": [5, 8, 10, 12],
    "simulation": [5, 8, 10, 12],
    "crc_stereocite": [5, 10, 15, 20, 25],
}
V3_MATCHED_FIELDS = (
    "epochs",
    "seed",
    "n_comps",
    "use_harmony",
    "spot_sampling",
    "max_spots_per_section",
    "max_shared_genes",
    "max_shared_peaks",
    "lambda_contrast",
    "hvg_num",
    "hvg_num_adt",
    "lr",
    "weight_decay",
    "ot_prior_mode",
    "bidirectional_ot_attention",
    "candidate_backend",
    "initial_modality_candidate_k",
    "candidate_k",
    "attention_topk",
    "faiss_nlist",
    "faiss_nprobe",
    "faiss_device",
    "faiss_train_sample_size",
    "faiss_query_batch_size",
    "uot_epsilon",
    "uot_tau_a",
    "uot_tau_b",
    "uot_stabilizer",
    "uot_max_iter",
    "update_interval",
    "spatial_knn_k",
    "graphsage_edge_batch_size",
    "decoder_chunk_size",
    "ot_attention_source_chunk_size",
    "training_loss_only",
    "checkpoint_ot_attention",
    "checkpoint_encoder_fusion",
    "checkpoint_decoder_chunks",
    "checkpoint_graph_encoder",
    "amp_dtype",
    "cache_spatial_graphs",
    "save_candidate_qc",
    "saved_embeddings",
    "saved_ot_prior_topk",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def gpu_check(python: str, label: str, attempts: int = 5) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
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
            cwd=V5_CODE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if smi.returncode == 0 and probe.returncode == 0:
            return {
                "checked_at": now(),
                "attempt": attempt,
                "nvidia_smi": smi.stdout.strip(),
                "torch_probe": probe.stdout.strip(),
            }
        failures.append(
            {
                "attempt": attempt,
                "nvidia_smi_returncode": smi.returncode,
                "nvidia_smi_stderr": smi.stderr.strip(),
                "torch_returncode": probe.returncode,
                "torch_stderr": probe.stderr.strip(),
            }
        )
        if attempt < attempts:
            time.sleep(5)
    raise RuntimeError(
        f"GPU unavailable for {label} after {attempts} checks: {failures}"
    )


def training_commands(python: str) -> list[tuple[str, list[str]]]:
    scripts = V5_CODE_ROOT / "scripts"
    output = RESULT_ROOT / "crc_stereocite" / VARIANT
    return [
        (
            "train_human_lymph_node",
            [python, str(scripts / "run_human_lymph_node.py")],
        ),
        (
            "train_mouse_spleen",
            [python, str(scripts / "run_mouse_spleen.py")],
        ),
        (
            "train_mouse_thymus",
            [python, str(scripts / "run_mouse_thymus.py")],
        ),
        (
            "train_simulation",
            [python, str(scripts / "run_simulation.py")],
        ),
        (
            "train_crc_stereocite",
            [
                python,
                str(scripts / "run_crc_stereocite.py"),
                "--data_dir", str(ROOT / "data/CRC_Stereo-CITE-seq"),
                "--output_dir", str(output),
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
                "--dynamic_candidate_source", "final",
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
    ]


def postprocess_commands(python: str) -> list[tuple[str, list[str]]]:
    return [
        (
            "analyze_small_datasets",
            [
                python,
                str(ROOT / "scripts/analyze_result_v3_standardized.py"),
                "--result-root", str(RESULT_ROOT),
                "--datasets",
                "human_lymph_node,mouse_spleen,mouse_thymus,simulation",
            ],
        ),
        (
            "analyze_crc_stereocite",
            [
                python,
                str(ROOT / "scripts/analyze_result_v3_large_standardized.py"),
                "--result-root", str(RESULT_ROOT),
            ],
        ),
    ]


def summary_path(task_name: str) -> Path:
    dataset = task_name.removeprefix("train_")
    return RESULT_ROOT / dataset / VARIANT / "run_summary.json"


def validate_training(task_name: str) -> dict[str, Any]:
    candidate_path = summary_path(task_name)
    baseline_path = (
        ROOT
        / "result_v3"
        / task_name.removeprefix("train_")
        / VARIANT
        / "run_summary.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"result_v3": baseline.get(key), "result_v5": candidate.get(key)}
        for key in V3_MATCHED_FIELDS
        if baseline.get(key) != candidate.get(key)
    }
    if mismatches:
        raise ValueError(f"{task_name}: non-model settings differ from v3: {mismatches}")
    model_mismatches = {}
    if candidate.get("ot_updates") != EXPECTED_OT_UPDATES:
        model_mismatches["ot_updates"] = candidate.get("ot_updates")
    if candidate.get("dynamic_candidate_source") != "final":
        model_mismatches["dynamic_candidate_source"] = candidate.get(
            "dynamic_candidate_source"
        )
    if bool(candidate.get("attention_context_gate_enabled", False)):
        model_mismatches["attention_context_gate_enabled"] = True
    if model_mismatches:
        raise ValueError(f"{task_name}: invalid v5 model settings: {model_mismatches}")
    trace_path = candidate_path.parent / "cuda_memory_trace.jsonl"
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise ValueError(f"{task_name}: missing CUDA memory trace: {trace_path}")
    return {
        "summary": str(candidate_path),
        "v3_baseline": str(baseline_path),
        "v3_matched_fields": list(V3_MATCHED_FIELDS),
        "v5_model_difference": {
            "first_dynamic_ot_refresh_epoch": 100,
            "ot_updates": EXPECTED_OT_UPDATES,
        },
        "cuda_memory_trace": str(trace_path),
    }


def validate_analysis() -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for dataset, expected in EXPECTED_RETAINED_K.items():
        path = (
            RESULT_ROOT
            / dataset
            / VARIANT
            / "analysis/standardized_embedding/config.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        actual = (
            config.get("retained_clustering_k_values")
            or config.get("plot_k_values")
            or config.get("plot_and_independent_k_values")
            or config.get("k_values")
        )
        if actual != expected:
            raise ValueError(
                f"{dataset}: clustering K mismatch: expected={expected}, actual={actual}"
            )
        config.update(
            {
                "model_variant": (
                    "v5_bidirectional_sparse_uot_fixed_lc0.1_delayed_ot_refresh"
                ),
                "first_dynamic_ot_refresh_epoch": 100,
                "ot_updates": EXPECTED_OT_UPDATES,
                "v3_matched_clustering_k_values": expected,
            }
        )
        write_json(path, config)
        validated[dataset] = {"config": str(path), "retained_k": actual}
    write_json(RESULT_ROOT / "v3_matched_clustering_validation.json", validated)
    return validated


def launch_daemon(python: str, from_task: str | None) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [python, str(Path(__file__).resolve()), "--python", python]
    if from_task:
        command.extend(["--from-task", from_task])
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
                "exit_file": str(EXIT_PATH),
                "supervisor_log": str(SUPERVISOR_LOG),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_suite(python: str, from_task: str | None) -> None:
    tasks = [
        *((name, command, True) for name, command in training_commands(python)),
        *((name, command, False) for name, command in postprocess_commands(python)),
    ]
    task_names = [name for name, _, _ in tasks]
    if from_task:
        if from_task not in task_names:
            raise ValueError(f"Unknown --from-task {from_task!r}: {task_names}")
        tasks = tasks[task_names.index(from_task) :]
    new_status = {
        "suite": "result_v5_missing_datasets_v3_matched_non_model_settings",
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "python": python,
        "v5_code_root": str(V5_CODE_ROOT),
        "v5_commit": V5_COMMIT,
        "datasets": [
            "Human_Lymph_Node",
            "Mouse_Spleen",
            "Mouse_Thymus",
            "Simulation",
            "CRC_Stereo-CITE-seq",
        ],
        "comparison_baseline": "result_v3",
        "comparison_policy": (
            "All preprocessing/training/analysis settings match result_v3; "
            "only the v5 first dynamic OT refresh changes from epoch 20 to 100."
        ),
        "expected_ot_updates": EXPECTED_OT_UPDATES,
        "expected_retained_clustering_k": EXPECTED_RETAINED_K,
        "status": "running",
        "tasks": {},
    }
    status = (
        json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if from_task and STATUS_PATH.is_file()
        else new_status
    )
    status["supervisor_pid"] = os.getpid()
    status["status"] = "running"
    status.pop("failed_task", None)
    status.pop("finished_at", None)
    write_json(STATUS_PATH, status)
    EXIT_PATH.unlink(missing_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        for name, command, needs_gpu in tasks:
            log_path = LOG_ROOT / f"{name}.log"
            task = {
                "status": "checking_gpu" if needs_gpu else "running",
                "started_at": now(),
                "command": command,
                "log": str(log_path),
            }
            status["tasks"][name] = task
            write_json(STATUS_PATH, status)
            if needs_gpu:
                task["gpu_check"] = gpu_check(python, name)
                task["status"] = "running"
                write_json(STATUS_PATH, status)
            print(f"[{now()}] START {name}", flush=True)
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=V5_CODE_ROOT if needs_gpu else ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=environment,
                )
            task.update(
                {
                    "status": "completed" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "finished_at": now(),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if result.returncode == 0 and needs_gpu:
                task["validation"] = validate_training(name)
            write_json(STATUS_PATH, status)
            if result.returncode != 0:
                raise RuntimeError(f"{name} failed with return code {result.returncode}")
            print(f"[{now()}] COMPLETED {name}", flush=True)
        status["analysis_validation"] = validate_analysis()
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "failed_task": name,
                "error": repr(error),
                "finished_at": now(),
            }
        )
        write_json(STATUS_PATH, status)
        write_json(EXIT_PATH, {"exit_code": 1, "finished_at": now(), "error": repr(error)})
        raise
    status.update({"status": "completed", "finished_at": now()})
    write_json(STATUS_PATH, status)
    write_json(EXIT_PATH, {"exit_code": 0, "finished_at": now()})
    print("RESULT_V5_MISSING_DATASETS_SUITE: PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python", default="/home/hujinlan/miniconda3/envs/cosie/bin/python"
    )
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--from-task", default=None)
    args = parser.parse_args()
    if args.daemon:
        launch_daemon(args.python, args.from_task)
    else:
        run_suite(args.python, args.from_task)


if __name__ == "__main__":
    main()
