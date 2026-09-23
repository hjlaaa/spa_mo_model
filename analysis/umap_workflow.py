"""UMAP projection/output orchestration over already selected, aligned inputs.

No input reader, method dispatch, scaler fitting or cohort selection. Scientific
parameters and coordinate caching stay with the existing analysis.umap kernel.
"""
from __future__ import annotations
from analysis.plotting import plot_individual, meaningful_biology_mask


def project_umap_views(views, args, *, source, force_float32, n_obs=None):
    """Yield one cached/fitted projection per supplied view in its existing order.

    Each view is (key, values, cache_path, validation_name). None preserves the
    external-method path's existing upstream validation rather than adding one.
    Lazy iteration preserves input-modality validation/failure timing.
    """
    from analysis.umap import fit_umap, validate_features
    for key, values, cache_path, validation_name in views:
        if validation_name is not None:
            validate_features(validation_name, values, n_obs)
        yield key, fit_umap(values, cache_path, args.n_neighbors, args.min_dist,
                            args.seed, args.overwrite_umap, source_identity=source,
                            force_float32=force_float32)


def save_umap_coordinate_table(metadata, coordinates, joint_labels, path, *, input_coordinates=None):
    """Original metadata-copy/column assembly and gzip CSV; never select rows."""
    table = metadata.copy()
    if input_coordinates is not None:
        for modality, values in input_coordinates.items():
            key = modality.lower()
            table[f"input_{key}_UMAP1"] = values[:, 0]
            table[f"input_{key}_UMAP2"] = values[:, 1]
    table["integrated_UMAP1"] = coordinates[:, 0]
    table["integrated_UMAP2"] = coordinates[:, 1]
    for k, labels in sorted(joint_labels.items()):
        table[f"joint_k{k}"] = labels
    table.to_csv(path, index=False, compression="gzip")


def run_method_projection(data, reference, args, *, source, method_name,
                          plot_panel, custom_biology_colors):
    """Project once, pass coordinates to the original plots, then save the table.

    The two supplied operations preserve the existing panel and optional color
    lookup (including its failure timing); they are not lifecycle callbacks.
    Config/README/source payloads remain owned by the caller.
    """
    figures_dir = data.output_dir / 'figures'
    tables_dir = data.output_dir / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    coordinates = next(project_umap_views(
        [('integrated', data.embedding, tables_dir / 'integrated_embedding_umap.npy', None)],
        args, source=source, force_float32=False))[1]
    plot_panel(data, reference, coordinates, figures_dir, args.seed)
    point_size = 2.0 if len(coordinates) >= 50000 else 8.0
    section_values = reference.metadata['section_label'].astype(str).to_numpy()
    biological_values = reference.metadata[reference.primary_label].to_numpy()
    biology_mask = meaningful_biology_mask(reference)
    include_biology = bool(biology_mask.any())
    plot_individual(figures_dir / 'integrated_by_section.png', coordinates, section_values, f"{method_name} — {reference.config['display_name']} integrated UMAP by section", 'section', point_size, args.seed)
    biological_path = figures_dir / f'integrated_by_{reference.primary_label}.png'
    if include_biology:
        plot_individual(biological_path, coordinates, biological_values, f"{method_name} — {reference.config['display_name']} integrated UMAP by {reference.primary_label}", 'biology', point_size, args.seed, custom_biology_colors(data, reference))
    else:
        biological_path.unlink(missing_ok=True)
    cluster_dir = figures_dir / 'integrated_joint_clusters'
    for (k, labels) in sorted(data.joint_labels.items()):
        plot_individual(cluster_dir / f'integrated_by_joint_k{k}.png', coordinates, labels.astype(str), f"{method_name} — {reference.config['display_name']} integrated UMAP by joint k={k}", 'cluster', point_size, args.seed)
    table_path = tables_dir / 'umap_coordinates_and_labels.csv.gz'
    save_umap_coordinate_table(reference.metadata, coordinates, data.joint_labels, table_path)
    return table_path, biological_path, biology_mask, include_biology
