"""Pure job planning and input checks for the removable v15C-2 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).parent / "configs/suite.json"


def load_suite(path=CONFIG):
    suite = json.loads(Path(path).read_text())
    if suite["schema_version"] != 1 or len(suite["datasets"]) != 6:
        raise ValueError("Expected the six-dataset v15C-2 suite.")
    if len(suite["seeds"]) != 3 or len(set(suite["seeds"])) != 3:
        raise ValueError("Expected three distinct training seeds.")
    return suite


def output_root(suite):
    return (ROOT / suite["output_root"]).resolve()


def render_job(suite, dataset, seed, stage, attempt=1, trained_dir=None):
    base = output_root(suite) / dataset["id"] / f"seed_{seed}"
    train_dir = base / f"train_attempt_{attempt:02d}"
    if dataset["train_subdir"]:
        train_dir /= dataset["train_subdir"]
    if trained_dir is not None:
        train_dir = Path(trained_dir)
    analysis_dir = base / f"analysis_attempt_{attempt:02d}"
    values = {"repo": str(ROOT), "seed": seed, "train_dir": str(train_dir),
              "analysis_dir": str(analysis_dir / "standardized_embedding" if dataset["id"] == "spatch" else analysis_dir)}
    script = dataset["runner"] if stage == "train" else "scripts/evaluate.py"
    argv = [str(suite["python"]), str(ROOT / script)]
    argv += [arg.format(**values) for arg in dataset[f"{stage}_argv"]]
    if stage == "analysis":
        argv += ["--writable-result-root", str(output_root(suite))]
    return {"id": f"{dataset['id']}/seed_{seed}/{stage}", "dataset": dataset["id"],
            "seed": seed, "stage": stage, "attempt": attempt, "command": argv,
            "output_dir": str(train_dir if stage == "train" else analysis_dir),
            "run_dir": str(train_dir),
            "depends_on": None if stage == "train" else f"{dataset['id']}/seed_{seed}/train"}


def jobs(suite):
    # Each repetition visits all six datasets, so an interruption does not leave
    # every result concentrated in the first dataset.
    return [render_job(suite, ds, seed, stage)
            for seed in suite["seeds"] for ds in suite["datasets"]
            for stage in ("train", "analysis")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(suite):
    files = []
    for folder in ("model", "training", "data_io", "analysis", "scripts", "experiments/v15c2"):
        files.extend((ROOT / folder).rglob("*.py"))
    files.extend((ROOT / "experiments/v15c2/configs").glob("*.json"))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}
    payload = json.dumps({"suite": suite, "files": hashes}, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "files": hashes}


def check_inputs(suite):
    """Only stat files and read small provenance; never load data arrays."""
    if not Path(suite["python"]).is_file():
        raise FileNotFoundError(suite["python"])
    if output_root(suite) != ROOT / "result_v15C-2":
        raise ValueError("This experiment writes only result_v15C-2.")
    for ds in suite["datasets"]:
        train_args = ds["train_argv"]
        if "--device" not in train_args or train_args[train_args.index("--device") + 1] != "cuda":
            raise ValueError(f"{ds['id']}: v15C-2 requires GPU training (--device cuda).")
        if ds["id"] == "spatch":
            flag = "--preprocessed_cache_dir"
            expected_cache = ROOT / "preprocessed_cache/spatch/v6_harmony_n50_hvg3000"
            if flag not in train_args or Path(train_args[train_args.index(flag) + 1]).resolve() != expected_cache:
                raise ValueError("SPATCH must reuse the historical model-ready preprocessing cache.")
            if "--no-build_preprocessed_cache" not in train_args or "--build_preprocessed_cache" in train_args:
                raise ValueError("SPATCH preprocessing cache must not be rebuilt.")
        for key in ("baseline_summary", "baseline_analysis_config"):
            if sha256(ROOT / ds[key]) != ds[key + "_sha256"]:
                raise ValueError(f"Historical source changed: {ds[key]}")
        for stage in ("train", "analysis"):
            command = render_job(suite, ds, suite["seeds"][0], stage)["command"]
            for flag in ("--data_dir", "--data-dir", "--config", "--asw-sample", "--preprocessed_cache_dir"):
                if flag not in command:
                    continue
                path = Path(command[command.index(flag) + 1])
                if not path.exists():
                    raise FileNotFoundError(f"{ds['id']} {flag}: {path}")
                if flag == "--config":
                    config = json.loads(path.read_text())
                    for section in config["sections"]:
                        for key, value in section.items():
                            if key.endswith("_input") and not Path(value).is_file():
                                raise FileNotFoundError(value)
                if flag == "--preprocessed_cache_dir" and not (path / "manifest.json").is_file():
                    raise FileNotFoundError(path / "manifest.json")
                if flag == "--preprocessed_cache_dir" and sha256(path / "manifest.json") != ds["cache_manifest_sha256"]:
                    raise ValueError("SPATCH cache manifest differs from historical v15C.")
    if sha256(ROOT / suite["baseline_suite"]) != suite["baseline_suite_sha256"]:
        raise ValueError("Historical suite source changed.")
    count = len(suite["datasets"]) * len(suite["seeds"])
    return {"datasets": len(suite["datasets"]), "repetitions": len(suite["seeds"]),
            "training_jobs": count, "analysis_jobs": count, "training_device": "cuda",
            "input_paths": "present", "gpu_probe": "not_performed", "data_loaded": False}
