#!/usr/bin/env python3
"""Complete missing result_v6 datasets with v3-matched non-model settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "result_v6"
LOG_ROOT = RESULT_ROOT / "logs"
STATUS_PATH = RESULT_ROOT / "missing_v3_matched_suite_status.json"
PID_PATH = RESULT_ROOT / "missing_v3_matched_suite.pid"
EXIT_PATH = RESULT_ROOT / "missing_v3_matched_suite.exit"
SUPERVISOR_LOG = LOG_ROOT / "missing_v3_matched_suite_supervisor.log"
VARIANT = "bidirectional_sparse_uot_fixed_lc0.1_seed42"
EXPECTED_OT_UPDATES = [100, 120, 140, 160, 180, 200]
EXPECTED_ARCHITECTURE = (
    "MLP+pre_OT_GraphSAGE+OT_attention+post_OT_GraphSAGE+MLP_decoder"
)
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
    errors: list[dict[str, Any]] = []
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
            return {
                "checked_at": now(),
                "attempt": attempt,
                "nvidia_smi": smi.stdout.strip(),
                "torch_probe": probe.stdout.strip(),
            }
        errors.append(
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
        f"GPU unavailable for {label} after {attempts} checks: {errors}"
    )


def training_commands(python: str) -> list[tuple[str, list[str]]]:
    def output(dataset: str) -> str:
        return str(RESULT_ROOT / dataset / VARIANT)

    return [
        (
            "train_human_lymph_node",
            [
                python,
                str(ROOT / "scripts/run_human_lymph_node.py"),
                "--output_dir", output("human_lymph_node"),
                "--dynamic_candidate_source", "ot",
                "--update_interval", "20",
            ],
        ),
        (
            "train_mouse_spleen",
            [
                python,
                str(ROOT / "scripts/run_mouse_spleen.py"),
                "--output_dir", output("mouse_spleen"),
                "--dynamic_candidate_source", "ot",
                "--update_interval", "20",
            ],
        ),
        (
            "train_mouse_thymus",
            [
                python,
                str(ROOT / "scripts/run_mouse_thymus.py"),
                "--output_dir", output("mouse_thymus"),
                "--dynamic_candidate_source", "ot",
                "--update_interval", "20",
            ],
        ),
        (
            "train_simulation",
            [
                python,
                str(ROOT / "scripts/run_simulation.py"),
                "--output_dir", output("simulation"),
                "--dynamic_candidate_source", "ot",
                "--update_interval", "20",
            ],
        ),
        (
            "train_crc_stereocite",
            [
                python,
                str(ROOT / "scripts/run_crc_stereocite.py"),
                "--data_dir", str(ROOT / "data/CRC_Stereo-CITE-seq"),
                "--output_dir", output("crc_stereocite"),
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
                "--dynamic_candidate_source", "ot",
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
    umap_script = str(ROOT / "scripts/generate_result_v6_spamosaic_umap.py")
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
        *[
            (
                f"umap_{dataset}",
                [python, umap_script, "--dataset", dataset],
            )
            for dataset in (
                "human_lymph_node",
                "mouse_spleen",
                "mouse_thymus",
                "simulation",
                "crc_stereocite",
            )
        ],
    ]


def summary_path_for_task(task_name: str) -> Path:
    dataset = task_name.removeprefix("train_")
    return RESULT_ROOT / dataset / VARIANT / "run_summary.json"


def validate_training(task_name: str) -> dict[str, Any]:
    summary_path = summary_path_for_task(task_name)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_path = (
        ROOT
        / "result_v3"
        / task_name.removeprefix("train_")
        / VARIANT
        / "run_summary.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"result_v3": baseline.get(key), "result_v6": summary.get(key)}
        for key in V3_MATCHED_FIELDS
        if baseline.get(key) != summary.get(key)
    }
    if mismatches:
        raise ValueError(f"{task_name}: non-model settings differ from v3: {mismatches}")
    expected_model = {
        "architecture": EXPECTED_ARCHITECTURE,
        "pre_post_graphsage_parameter_sharing": False,
        "ot_refresh_embedding_key": "ot_embeddings",
        "dynamic_candidate_source": "ot",
        "attention_context_gate_enabled": False,
        "ot_updates": EXPECTED_OT_UPDATES,
    }
    model_mismatches = {
        key: {"expected": value, "actual": summary.get(key)}
        for key, value in expected_model.items()
        if summary.get(key) != value
    }
    if model_mismatches:
        raise ValueError(f"{task_name}: invalid v6 model settings: {model_mismatches}")
    trace_path = summary_path.parent / "cuda_memory_trace.jsonl"
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise ValueError(f"{task_name}: CUDA memory trace missing or empty: {trace_path}")
    return {
        "summary": str(summary_path),
        "baseline": str(baseline_path),
        "v3_matched_fields": list(V3_MATCHED_FIELDS),
        "v6_model_settings": expected_model,
        "cuda_memory_trace": str(trace_path),
    }


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
        "suite": "result_v6_missing_datasets_v3_matched_non_model_settings",
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "python": python,
        "datasets": [
            "Human_Lymph_Node",
            "Mouse_Spleen",
            "Mouse_Thymus",
            "Simulation",
            "CRC_Stereo-CITE-seq",
        ],
        "comparison_baseline": "result_v3",
        "comparison_policy": (
            "Match v3 non-model settings; retain v6 dual-GraphSAGE architecture, "
            "ot_embeddings refresh source, and delayed first OT refresh at epoch 100."
        ),
        "expected_ot_updates": EXPECTED_OT_UPDATES,
        "expected_architecture": EXPECTED_ARCHITECTURE,
        "expected_pre_post_graphsage_parameter_sharing": False,
        "expected_ot_refresh_embedding_key": "ot_embeddings",
        "expected_dynamic_candidate_source": "ot",
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
            if result.returncode == 0 and needs_gpu:
                task["validation"] = validate_training(name)
            write_json(STATUS_PATH, status)
            if result.returncode != 0:
                raise RuntimeError(f"{name} failed with return code {result.returncode}")
            print(f"[{now()}] COMPLETED {name}", flush=True)
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
    print(f"[{now()}] RESULT_V6_MISSING_DATASETS_SUITE: PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--from-task", default=None)
    args = parser.parse_args()
    if args.daemon:
        launch_daemon(args.python, args.from_task)
    else:
        run_suite(args.python, args.from_task)


if __name__ == "__main__":
    main()
