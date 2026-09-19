"""Explicit single-run evaluation, using the accepted loaders and protocols."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from analysis import loaders, protocols
from analysis.cache import (begin_analysis, finish_analysis, check_output_path, data_identity,
                            file_identity, implementation_identity)
from analysis.metrics import compute_joint_per_section_supervised_metrics
from analysis.plotting import plot_dataset_spatial
from analysis.requested import analyze_prepared, truth_arrays

DATASETS = ("mousebrain", "human_embryo", "misar_seq", "spatch", "crc_stereocite",
            "human_lymph_node", "mouse_spleen", "mouse_thymus", "simulation")
DISPLAY_NAMES = {"mousebrain": "MouseBrain", "human_embryo": "Human Embryo", "misar_seq": "MISAR-seq",
                 "spatch": "SPATCH", "crc_stereocite": "CRC", "human_lymph_node": "Human Lymph Node",
                 "mouse_spleen": "Mouse Spleen", "mouse_thymus": "Mouse Thymus", "simulation": "Simulation"}


def load_dataset(dataset, run_dir, data_dir, output_dir, *, method_name):
    """Explicit routes into P6a. In particular, MISAR analysis keeps its old order."""
    kw = dict(name=method_name, output_root=output_dir)
    if dataset == "mousebrain":
        data = loaders.load_mousebrain(run_dir, data_dir, section_order=["s1", "s2", "s3"],
            section_dirs={"s1": "dataset_MouseBrain_SectionA", "s2": "dataset_MouseBrain_SectionB", "s3": "dataset_MouseBrain_SectionC"},
            label_keys=protocols.MOUSEBRAIN["LABEL_KEYS"], group_label="group", **kw)
    elif dataset == "misar_seq":
        data = loaders.load_misar(run_dir, data_dir, section_order=["dataset1", "dataset2", "dataset3", "dataset4"],
            label_keys=protocols.MISAR["LABEL_KEYS"], **kw)
    elif dataset == "human_lymph_node":
        data = loaders.load_paired_rna_adt(run_dir, data_dir, section_order=["Human_Lymph_Node_A1", "Human_Lymph_Node_D1"], **kw)
    elif dataset == "mouse_spleen":
        data = loaders.load_paired_rna_adt(run_dir, data_dir, section_order=["Mouse_Spleen1", "Mouse_Spleen2"], **kw)
    elif dataset == "mouse_thymus":
        data = loaders.load_thymus(run_dir, data_dir, section_order=[f"Mouse_Thymus{i}" for i in range(1, 5)], **kw)
    elif dataset == "simulation":
        data = loaders.load_simulation(run_dir, data_dir, section_order=[f"Simulation{i}" for i in range(1, 6)], **kw)
    elif dataset == "crc_stereocite":
        from analysis.comparisons import CRC_SECTIONS, CRC_SHORT_NAMES
        data = loaders.load_crc(run_dir, data_dir, section_order=CRC_SECTIONS,
                                short_names=CRC_SHORT_NAMES, output_root=output_dir)
        data["name"] = method_name
    elif dataset == "spatch":
        from analysis import spatch
        metadata = spatch.load_metadata(run_dir)
        data = dict(name=method_name, embedding=spatch.load_embedding(run_dir),
                    sections=metadata["section"].to_numpy(), barcodes=metadata["spot_id"].to_numpy(),
                    coords=metadata[["x", "y"]].to_numpy(), metadata=metadata,
                    truth={label: metadata[label].to_numpy() for label in spatch.TRUTH_LABELS},
                    source_paths=[run_dir / f"final_embeddings_{s}.npy" for s in spatch.SECTIONS] + [run_dir / "spot_metadata.csv.gz"],
                    data_dir=run_dir, output_root=output_dir, shared_sources={})
    elif dataset == "human_embryo":
        from analysis.inputs import load_human_embryo
        summary, sections, embeddings, metadata = load_human_embryo(run_dir)
        # The original loader's historical identity limitation is recorded, not invented.
        ids = []
        for section in sections:
            frame = metadata[section]
            key = next((k for k in ("obs_name", "cellid", "original_row") if k in frame), None)
            if key is None:
                raise ValueError("This embedding-only format has no proven spot identity for this evaluation; use embryo-reference with its original loader contract")
            ids.extend(frame[key].astype(str).tolist())
        data = dict(name=method_name, embedding=np.vstack([embeddings[s] for s in sections]),
                    sections=np.concatenate([np.repeat(s, len(embeddings[s])) for s in sections]), barcodes=np.asarray(ids),
                    coords=np.vstack([metadata[s][["x", "y"]].to_numpy() for s in sections]),
                    truth={"celltype": np.concatenate([metadata[s]["celltype"].astype(str).to_numpy() for s in sections])},
                    source_paths=[Path(summary["saved_files"]["embeddings"][s]) for s in sections] +
                                 [Path(summary["preprocess_manifest"]["annotation_files"][s]) for s in sections],
                    data_dir=run_dir, output_root=output_dir, shared_sources={})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return SimpleNamespace(**data)


def supervised_truth(data, dataset):
    if dataset in {"spatch", "human_embryo"}:
        return data.truth
    return truth_arrays(data, dataset)


def evaluate_joint_per_section(data, dataset, assignments_dir, output_dir, k_values, *, run_dir, protocol_name):
    """Consume saved joint assignments only; this path never calls clustering."""
    truth = supervised_truth(data, dataset)
    sections = list(dict.fromkeys(np.asarray(data.sections).astype(str)))
    protocol = protocols.joint_per_section_protocol(DISPLAY_NAMES[dataset], k_values, sections)
    loaded = {k: loaders.load_joint_assignments(assignments_dir, k, data.sections, data.barcodes)
              for k in k_values} if protocol["status"] == "applicable" else {}
    source = {"data": data_identity(data, truth), "protocol_name": protocol_name, "protocol": protocol,
              "assignments": {str(k): [file_identity(p) for p in files] for k, (_, files) in loaded.items()},
              "implementation": implementation_identity("joint-per-section-export-v1")}
    if begin_analysis(output_dir, source, input_dirs=[run_dir, *([assignments_dir] if loaded else [])]):
        return json.loads((output_dir / "summary.json").read_text())
    rows = []
    for k, (labels, files) in loaded.items():
        rows.extend(compute_joint_per_section_supervised_metrics(DISPLAY_NAMES[dataset], labels,
            data.sections, truth, k=k, section_order=sections,
            joint_label_source=";".join(str(p) for p in files),
            truth_source=str(output_dir / "analysis_source_manifest.json") + "#identity.data.truth")["rows"])
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "joint_per_section_supervised_metrics.csv", index=False)
    record = {"dataset": dataset, "protocol": protocol_name, "scope": "joint_per_section",
              "status": protocol["status"], "rows": len(rows), "aggregation": "none"}
    (output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    finish_analysis(output_dir, source)
    return record


def render_requested(data, dataset, analysis_dir, output_dir, *, k_values, modes, run_dir):
    """Render the original requested retained-K spatial plots from saved labels."""
    items = []
    for mode in modes:
        for k in k_values:
            for section in np.unique(data.sections):
                mask = np.asarray(data.sections) == section
                path = analysis_dir / "clustering" / f"{mode}_k{k}" / f"labels_{section}.csv"
                frame = pd.read_csv(path, dtype={"obs_name": str})
                if len(frame) != int(mask.sum()) or not np.array_equal(frame["obs_name"].to_numpy(), np.asarray(data.barcodes)[mask]):
                    raise ValueError(f"Plot label/spot order mismatch: {path}")
                items.append((mode, k, section, mask, frame["cluster"].to_numpy(dtype=int), path))
    source = {"data": data_identity(data, supervised_truth(data, dataset)),
              "assignments": [file_identity(item[-1]) for item in items],
              "protocol": {"dataset": dataset, "name": "requested", "modes": list(modes), "plot_k": list(k_values)},
              "implementation": implementation_identity("requested-spatial-plot-v1")}
    if begin_analysis(output_dir, source, input_dirs=[run_dir, analysis_dir]):
        return
    for mode, k, section, mask, labels, _ in items:
        directory = output_dir / "clustering" / f"{mode}_k{k}"
        directory.mkdir(parents=True, exist_ok=True)
        plot_dataset_spatial(dataset, directory / f"spatial_{section}.png", np.asarray(data.coords)[mask], labels,
            f"{data.name} standardized_embedding {mode} K={k} {section}", section=section, k=k)
    finish_analysis(output_dir, source)


def evaluate(args):
    """One explicit request; no training imports, suite globals or secondary parse."""
    run_dir = args.run_dir.resolve()
    if args.method_name is None:
        args.method_name = f"spa_mo_model_result_{args.model_version}" if args.protocol == "requested" else "spa_mo_model"
    if args.protocol == "requested" and args.preprocessing != "standardized_embedding":
        raise ValueError("Requested analysis uses standardized embedding; choose comparison for the raw protocol")
    output = check_output_path(args.output_dir, [run_dir])
    modes = tuple(mode for mode in ("joint", "independent") if mode in args.scope)
    per_section = args.joint_per_section or "joint_per_section" in args.scope
    if per_section and modes and "joint" not in modes:
        raise ValueError("joint_per_section requires joint scope, or use it alone with saved --assignments-dir")
    if per_section and args.preprocessing != "standardized_embedding":
        raise ValueError("The reference joint_per_section protocol uses joint standardized labels")
    if args.no_plots and not (args.protocol == "requested" and args.dataset in protocols.SPECS):
        raise ValueError("--no-plots is supported by the five generic requested workflows only")
    if args.plot_k is not None and not (args.protocol == "embryo-reference" or
            (args.protocol == "requested" and args.dataset in protocols.SPECS) or
            (args.protocol == "comparison" and args.dataset == "spatch")):
        raise ValueError("--plot-k is supported by generic requested, SPATCH comparison and embryo-reference")
    if args.protocol in {"umap-rerender", "embryo-per-section"} and (per_section or args.scope != ["joint", "independent"]):
        raise ValueError("Rendering uses saved assignments; --scope/--joint-per-section belong to metric protocols")
    if args.plot_seed is not None and args.protocol != "embryo-per-section":
        raise ValueError("--plot-seed is specific to embryo-per-section")
    if args.protocol == "umap-rerender":
        from analysis.umap import rerender
        config = args.umap_config or run_dir / "umap_config.json"
        if json.loads(config.read_text())["dataset"] != args.dataset:
            raise ValueError("UMAP config dataset differs from --dataset")
        return rerender(config, output)
    if args.protocol == "embryo-reference" and modes:
        if args.dataset != "human_embryo":
            raise ValueError("embryo-reference is specific to Human Embryo")
        from analysis import embryo
        k = protocols.EMBRYO["EXPERIMENT_K"] if args.k is None else args.k
        plot_k = k if args.plot_k is None else args.plot_k
        if modes != ("joint", "independent") and not args.skip_hierarchy:
            raise ValueError("A one-scope Embryo run requires --skip-hierarchy; the reference hierarchy uses both flat partitions")
        options = SimpleNamespace(input_dir=run_dir, output_dir=output,
            joint_k=k if "joint" in modes else [], independent_k=k if "independent" in modes else [],
            plot_k=plot_k, metric_sample_size=protocols.EMBRYO["metric_sample_size"],
            batch_metric_sample_size=protocols.EMBRYO["batch_metric_sample_size"],
            batch_asw_sample_size=protocols.EMBRYO["batch_asw_sample_size"],
            plot_sample_per_section=args.maximum_per_section, spatial_neighbor_k=6,
            batch_size=protocols.EMBRYO["batch_size"], n_init=protocols.EMBRYO["n_init"],
            max_iter=protocols.EMBRYO["max_iter"], seed=protocols.EMBRYO["seed"],
            reuse_cache=True, reuse_flat_labels=True, skip_hierarchy=args.skip_hierarchy)
        embryo.analyze(options, model_version=args.model_version)
        record = {"dataset": args.dataset, "analysis": str(output)}
        if per_section:
            data = load_dataset(args.dataset, run_dir, run_dir, output, method_name=args.method_name)
            record["joint_per_section"] = evaluate_joint_per_section(data, args.dataset, output,
                output / "joint_per_section", k, run_dir=run_dir, protocol_name=args.protocol)
        return record
    if args.protocol == "embryo-per-section":
        if args.dataset != "human_embryo" or args.assignments_dir is None:
            raise ValueError("embryo-per-section requires Human Embryo and --assignments-dir")
        from analysis.embryo import complete_per_section
        return complete_per_section(run_dir, args.maximum_per_section, protocols.EMBRYO["seed"] if args.plot_seed is None else args.plot_seed, False,
                                    output_root=output, analysis_dir=args.assignments_dir)
    if args.dataset not in {"spatch", "human_embryo"} and args.data_dir is None:
        raise ValueError("This loader requires explicit --data-dir for the existing metadata/annotation source")
    data_dir = run_dir if args.data_dir is None else args.data_dir.resolve()
    check_output_path(output, [run_dir, data_dir])
    if args.protocol == "requested" and args.dataset == "spatch" and modes:
        from analysis import spatch
        record = spatch.analyze(run_dir, args.model_version, args.post_ot_graphsage_scale,
                               output_dir=output, joint_per_section=per_section,
                               k_values=args.k, modes=modes)
        return record
    data = load_dataset(args.dataset, run_dir, data_dir, output, method_name=args.method_name)
    if not modes:
        if not per_section or args.assignments_dir is None or args.k is None:
            raise ValueError("joint_per_section alone requires --assignments-dir and --k; no clustering will be run")
        return evaluate_joint_per_section(data, args.dataset, args.assignments_dir, output, args.k,
                                         run_dir=run_dir, protocol_name=args.protocol)
    if args.protocol == "requested":
        if args.dataset not in protocols.SPECS:
            raise ValueError("Use comparison for CRC/HLN, or embryo-reference for Human Embryo; protocols are not interchangeable")
        spec = protocols.SPECS[args.dataset]
        spec = replace(spec,
            joint_ks=tuple(args.k or spec.joint_ks) if "joint" in modes else (),
            independent_ks=tuple(args.k or spec.independent_ks) if "independent" in modes else ())
        asw_mask = None
        if args.dataset == "mouse_thymus":
            if args.asw_sample is None:
                raise ValueError("Mouse Thymus requested requires its existing --asw-sample identity file")
            from analysis.sampling import key_mask
            asw_mask = key_mask(data, pd.read_csv(args.asw_sample, dtype=str))
        if not args.no_plots:
            available = set(spec.joint_ks if "joint" in modes else spec.independent_ks)
            if len(modes) == 2:
                available &= set(spec.independent_ks)
            plot_ks = args.plot_k if args.plot_k is not None else [k for k in protocols.REQUESTED_PLOT_KS[args.dataset] if k in available]
            if not set(plot_ks) <= available:
                raise ValueError("--plot-k must refer to fitted K in every selected scope")
        analysis_dir = output / "standardized_embedding"
        record = analyze_prepared(data, spec, args.model_version, args.post_ot_graphsage_scale,
                                  input_dir=run_dir, output_dir=analysis_dir,
                                  persisted_asw_mask=asw_mask, joint_per_section=per_section)
        if not args.no_plots:
            render_requested(data, args.dataset, analysis_dir, output / "figures", k_values=plot_ks,
                             modes=modes, run_dir=run_dir)
        return record
    if args.protocol == "comparison":
        return evaluate_comparison(data, args, run_dir, output, modes, per_section)
    raise ValueError(f"Unsupported protocol: {args.protocol}")


def evaluate_comparison(data, args, run_dir, output, modes, per_section):
    from analysis import comparisons, spatch
    dataset = args.dataset
    scheme = args.preprocessing
    if dataset == "crc_stereocite":
        k = comparisons.CRC_KS if args.k is None else args.k
        source = {"dataset": dataset, "data": data_identity(data, {}),
                  "protocol": {"name": "CRC comparison", "settings": protocols.CRC, "scheme": scheme,
                               "k": k, "modes": list(modes)},
                  "implementation": implementation_identity("crc-comparison-workflow-v1")}
        if begin_analysis(output, source, input_dirs=[run_dir]):
            return {"dataset": dataset, "analysis": str(output / scheme)}
        asw = comparisons.crc_select_by_barcode_hash(data, comparisons.CRC_ASW_SECTION_COUNTS, "asw_seed0")
        plots = comparisons.crc_select_by_barcode_hash(data, comparisons.CRC_PLOT_SECTION_COUNTS, "plot_seed0")
        graphs = comparisons.crc_prepare_shared(data, asw, plots)
        comparisons.crc_run_scheme(data, scheme, asw, plots, graphs,
            joint_ks=k if "joint" in modes else [], independent_ks=k if "independent" in modes else [])
        finish_analysis(output, source)
    elif dataset == "human_lymph_node":
        joint_k = comparisons.HLN_METRIC_KS if args.k is None else args.k
        independent_k = comparisons.HLN_PLOT_KS if args.k is None else args.k
        source = {"dataset": dataset, "data": data_identity(data, {}),
                  "protocol": {"name": "HLN comparison", "settings": protocols.HLN, "scheme": scheme,
                               "joint_k": joint_k, "independent_k": independent_k, "modes": list(modes)},
                  "implementation": implementation_identity("hln-comparison-workflow-v1")}
        if begin_analysis(output, source, input_dirs=[run_dir]):
            return {"dataset": dataset, "analysis": str(output / scheme)}
        comparisons.hln_run_scheme(data, scheme, joint_ks=joint_k if "joint" in modes else [],
                                   independent_ks=independent_k if "independent" in modes else [])
        finish_analysis(output, source)
    elif dataset == "spatch":
        if args.batch_metrics_source is None:
            raise ValueError("SPATCH comparison retains historical batch values; supply --batch-metrics-source explicitly")
        spec = SimpleNamespace(name=args.method_name, run_dir=run_dir, data_dir=run_dir,
            metadata_path=run_dir / "spot_metadata.csv.gz",
            embedding_paths={s: run_dir / f"final_embeddings_{s}.npy" for s in spatch.SECTIONS},
            output_root=output, batch_metrics_source=args.batch_metrics_source)
        source = spatch.source_identity(spec, data.metadata)
        # Shared identities are owned by a separate B9 boundary, not a preprocessing cache.
        shared = output / "shared_metrics"
        shared_source = {"source": source, "implementation": "spatch-shared-analysis-v1"}
        if begin_analysis(shared, shared_source, input_dirs=[run_dir]):
            graphs = {s: np.load(shared / f"spatial_neighbors_{s}.npz")["neighbor_indices"] for s in spatch.SECTIONS}
        else:
            graphs = spatch.prepare_shared(spec, data.metadata, None)
            finish_analysis(shared, shared_source)
        spatch.run_scheme(spec, data.metadata, data.embedding, graphs, scheme, source=source,
                          all_ks=args.k, retain_ks=args.plot_k, modes=modes)
        if per_section:
            if args.preprocessing != "standardized_embedding":
                raise ValueError("Reference joint_per_section uses joint standardized labels")
            k = args.k if args.k is not None else spatch.RETAIN_KS
            evaluate_joint_per_section(data, dataset, output / scheme, output / "joint_per_section", k,
                                      run_dir=run_dir, protocol_name="SPATCH compare")
    else:
        raise ValueError("This entry supports requested for this dataset; its cross-method comparison CLI remains documented")
    return {"dataset": dataset, "protocol": "comparison", "analysis": str(output / scheme),
            "joint_per_section": "not_applicable" if dataset != "spatch" and per_section else None}

