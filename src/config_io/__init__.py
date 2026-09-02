"""Shared YAML-to-frozen-dataclass config loading, used by every sub-project
that owns a `configs/*.yaml` (ppo, sequence_model, contrastive_pretrain,
pokemon_env). A leaf module: depends on nothing project-specific."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def load_dataclass_config[T: DataclassInstance](
    cls: type[T], path: str | Path, *, coerce: dict[str, Callable] | None = None
) -> T:
    """Loads `path` as YAML, rejects any key that isn't a field of `cls`, and
    constructs `cls(**data)`. `coerce` applies a per-field conversion (e.g.
    list -> tuple) before construction, for the rare field a dataclass wants
    in a type YAML can't express directly."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if coerce:
        for key, fn in coerce.items():
            if key in data:
                data[key] = fn(data[key])
    valid_fields = {f.name for f in fields(cls)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"unknown config field(s): {sorted(unknown)}")
    return cls(**data)
