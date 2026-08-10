"""
Reads a YAML file into a settings object, and complains clearly when it cannot.

Every run is described by one file in `configs/`. I set it up to check everything up
front rather than let a typo surface as a crash forty minutes into training, so an
unknown key, a missing key or a wrong type all fail straight away, naming the offender.
"""

from __future__ import annotations

import dataclasses
import difflib
import types
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

C = TypeVar("C")

_MISSING = dataclasses.MISSING


class ConfigError(ValueError):
    """Raised for a bad config: unknown key, missing key, wrong type, or bad value."""


# Schemas
@dataclass(frozen=True)
class PhaseConfig:
    """
    Everything a phase recognition run needs. This is what `configs/phase_*.yaml` fills in.

    Edit these values, not the ones in `PhaseModelConfig`. That class covers only the
    fields the network itself needs, and `PhaseModelConfig.from_config` copies them
    across from here when the model is built.

    Splits are whole videos. Frames from one video never land in two different splits.
    """

    data_root: str

    splits: str = ""
    train_split: tuple[str, ...] = ()
    val_split: tuple[str, ...] = ()
    test_split: tuple[str, ...] = ()

    backbone: str = "timm:convnext_tiny"
    pretrained: bool = True
    freeze_backbone: bool = False

    temporal: str = "tcn"
    temporal_ch: int = 256
    temporal_layers: int = 8
    n_heads: int = 4
    dropout: float = 0.1

    num_steps: int = 14
    num_instruments: int = 18
    instrument_loss_weight: float = 1.0
    instrument_threshold: float = 0.5

    clip_len: int = 64
    stride: int = 32
    img_size: int = 224

    lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    grad_clip: float = 1.0
    epochs: int = 30
    batch_size: int = 2
    workers: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        _one_of("temporal", self.temporal, ("tcn", "transformer"))
        _positive(self, "temporal_ch", "temporal_layers", "n_heads", "num_steps",
                  "num_instruments", "clip_len", "stride", "img_size", "epochs",
                  "batch_size", "lr")
        _non_negative(self, "dropout", "weight_decay", "grad_clip", "workers",
                      "instrument_loss_weight")
        if self.stride > self.clip_len:
            raise ConfigError(f"stride ({self.stride}) > clip_len ({self.clip_len}): "
                              "frames between windows would never be seen")
        if not 0.0 < self.instrument_threshold < 1.0:
            raise ConfigError(f"instrument_threshold must be in (0, 1), got "
                              f"{self.instrument_threshold}")


@dataclass(frozen=True)
class SegConfig:
    """
    Everything a segmentation run needs. This is what `configs/seg_*.yaml` fills in.

    `model: tinyunet` is the one that runs here and needs no downloads. `model: sam2`
    needs the SAM 2 package and a checkpoint file, neither of which is installed.
    """

    data_root: str

    model: str = "sam2"
    checkpoint: str = ""
    model_cfg: str = ""

    img_size: int = 1024
    prompt_kind: str = "point"

    splits: str = ""
    train_split: tuple[str, ...] = ()
    val_split: tuple[str, ...] = ()
    test_split: tuple[str, ...] = ()

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    train_decoder: bool = True

    lr: float = 5.0e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 40
    batch_size: int = 2
    workers: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        _one_of("model", self.model, ("sam2", "tinyunet"))
        _one_of("prompt_kind", self.prompt_kind, ("point", "box"))
        _positive(self, "img_size", "lora_rank", "lora_alpha", "epochs", "batch_size", "lr")
        _non_negative(self, "lora_dropout", "weight_decay", "grad_clip", "workers")
        if self.model == "sam2" and not (self.checkpoint and self.model_cfg):
            raise ConfigError("model: sam2 requires both `checkpoint` and `model_cfg` "
                              "(use model: tinyunet for a checkpoint-free run)")


def load_config(path: str | Path, schema: type[C],
                overrides: Mapping[str, Any] | None = None) -> C:
    """Read a YAML file into `schema`. Anything in `overrides` wins over the file."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {p}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: not valid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping, got {type(raw).__name__}")
    if overrides:
        raw = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
    return config_from_mapping(raw, schema, source=str(p))


def config_from_mapping(data: Mapping[str, Any], schema: type[C],
                        source: str = "<mapping>") -> C:
    """Build a settings object from a plain dict, checking every key and type."""
    known = {f.name: f for f in fields(schema)}  # type: ignore[arg-type]
    hints = typing.get_type_hints(schema)

    unknown = [k for k in data if k not in known]
    if unknown:
        key = unknown[0]
        near = difflib.get_close_matches(key, known, n=1)
        hint = f" (did you mean {near[0]!r}?)" if near else ""
        raise ConfigError(f"{source}: unknown key {key!r} for {schema.__name__}{hint}. "
                          f"Known keys: {', '.join(sorted(known))}")

    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name in data:
            kwargs[name] = _coerce(data[name], hints[name], name, source)
        elif f.default is _MISSING and f.default_factory is _MISSING:  # type: ignore[misc]
            raise ConfigError(f"{source}: missing required key {name!r} "
                              f"({_type_name(hints[name])}) for {schema.__name__}")
    try:
        return schema(**kwargs)  # type: ignore[call-arg]
    except ConfigError as exc:
        raise ConfigError(f"{source}: {exc}") from None


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Back to a plain dict, ready to save. `config_from_mapping` reads it back in."""
    return {k: list(v) if isinstance(v, tuple) else v
            for k, v in dataclasses.asdict(cfg).items()}


def _type_name(ann: Any) -> str:
    return getattr(ann, "__name__", None) or str(ann).replace("typing.", "")


def _coerce(value: Any, ann: Any, key: str, source: str) -> Any:
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return _coerce(value, inner[0], key, source)
        raise ConfigError(f"{source}: key {key!r} has unsupported schema type {_type_name(ann)}")

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{source}: key {key!r} must be a list, "
                              f"got {type(value).__name__}")
        item_ann = args[0] if args else Any
        return tuple(_coerce(v, item_ann, f"{key}[{i}]", source)
                     for i, v in enumerate(value))

    if ann is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{source}: key {key!r} must be a bool (true/false), "
                              f"got {value!r}")
        return value
    if ann is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{source}: key {key!r} must be an int, got {value!r}")
        return value
    if ann is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{source}: key {key!r} must be a number, got {value!r}. "
                              "YAML needs an explicit exponent form like 1.0e-4, "
                              "not 1e-4 (which parses as a string).")
        return float(value)
    if ann is str:
        if not isinstance(value, str):
            raise ConfigError(f"{source}: key {key!r} must be a string, got {value!r}")
        return value
    if ann is Any:
        return value
    raise ConfigError(f"{source}: key {key!r} has unsupported schema type {_type_name(ann)}")


def _one_of(key: str, value: Any, allowed: tuple[Any, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{key} must be one of {allowed}, got {value!r}")


def _positive(cfg: Any, *keys: str) -> None:
    for k in keys:
        if getattr(cfg, k) <= 0:
            raise ConfigError(f"{k} must be > 0, got {getattr(cfg, k)!r}")


def _non_negative(cfg: Any, *keys: str) -> None:
    for k in keys:
        if getattr(cfg, k) < 0:
            raise ConfigError(f"{k} must be >= 0, got {getattr(cfg, k)!r}")


__all__ = ["ConfigError", "PhaseConfig", "SegConfig", "load_config",
           "config_from_mapping", "config_to_dict"]
