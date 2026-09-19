"""Resolve ordinary config dictionaries before constructing a model.

Dataset defaults and CLI-to-field mappings stay in their entrypoints. This
module does not load data, construct models, run training, or write artifacts.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from model.configure import get_default_model_config, reject_unsupported_model_config


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_config(base: dict[str, Any], updates: Mapping[str, Any] | None):
    """Merge into base, preserving the existing recursive dictionary semantics."""
    if updates is None:
        return base
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def apply_explicit_overrides(config: dict[str, Any], overrides: Mapping[str, Any] | None):
    """None means absent; explicit False, zero and empty values are retained."""
    if overrides is None:
        return config
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            apply_explicit_overrides(config.setdefault(key, {}), value)
        elif value is not None:
            config[key] = value
    return config


def resolve_model_config(*, base=None, model_config=None, input_config=None, explicit_overrides=None):
    """Resolve defaults < preset/model < dataset/input < explicit CLI.

    All layers use the model dictionary's existing field hierarchy. Runners
    select their model/training layers from dataset JSON; dataset metadata is
    not subjected to model-field rejection.
    """
    resolved = get_default_model_config() if base is None else deepcopy(base)
    for layer in (model_config, input_config):
        if layer is not None:
            reject_unsupported_model_config(layer)
            merge_config(resolved, deepcopy(layer))
    if explicit_overrides is not None:
        reject_unsupported_model_config(explicit_overrides)
        apply_explicit_overrides(resolved, deepcopy(explicit_overrides))
    return resolved


def resolve_option_values(section, cli_values, defaults):
    """Resolve runner options already expressed as explicit values or None."""
    values = {}
    for key, default in defaults.items():
        value = cli_values[key]
        if value is None:
            value = section.get(key, default)
        section[key] = value
        values[key] = value
    return values


def config_source_layers(*, default, model_config=None, dataset_input=None, explicit_cli=None):
    """Describe source layers separately from the effective model dictionary."""
    return [
        {"layer": name, "source": deepcopy(source)}
        for name, source in (
            ("default", default),
            ("preset/model config", model_config),
            ("dataset/input config", dataset_input),
            ("explicit CLI", explicit_cli),
        )
    ]


def serialize_config(config):
    """Serialize JSON config values without tensor/array coercion or file I/O."""
    return json.dumps(config, indent=2, ensure_ascii=False)


def parse_dataset_args(parser, argv, defaults):
    """Parse user input once, then fill only absent dataset options.

    Parser actions use None for absence, including booleans. Dataset defaults
    stay in the owning runner; this helper does not choose hyperparameters.
    """
    args = parser.parse_args(argv)
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, deepcopy(value))
    return args


def describe_run_config(args, model_config, *, dataset, section_order, modalities, preprocessing):
    """Record the already resolved objects consumed by this run; never merge.

    The runner dictionary is the same Namespace storage used by the helpers.
    Data-derived section identity can be filled after reading a manifest.
    """
    return {
        "dataset": dataset,
        "section_order": section_order,
        "modalities": modalities,
        "model_config": model_config,
        "runner": vars(args),
        "preprocessing": preprocessing,
        "output_dir": str(Path(args.output_dir)),
    }
