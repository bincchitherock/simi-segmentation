"""
Which videos go in train, which in val, and which in test.

The three groups have different jobs. Train is what the model learns from. Val picks
which saved copy of the model to keep. Test is looked at once, at the very end.

I split by whole video, never by frame. Neighbouring frames of one video looked almost
identical, so putting some in train and some in test would have let the model recognise
what it already saw and score far too well. I wrote everything here to make that mistake
loud instead of silent, so overlapping groups raise an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SPLITS_FILENAME = "splits.json"

SPLIT_PROVENANCE: dict[str, str] = {
    "train": "the split the model was fitted on",
    "val": "the split the checkpoint was selected on, not a held-out estimate",
    "test": "held out from both training and checkpoint selection",
}


class SplitError(ValueError):
    """Raised when a split is missing, empty, overlapping, or names an unknown video."""


def assign_splits(ids: Sequence[str], sizes: Mapping[str, int]) -> dict[str, list[str]]:
    """Sort the ids and hand out the first `n` to each group, in order."""
    ordered = sorted(ids)
    if len(ordered) != len(set(ordered)):
        raise SplitError("duplicate ids")
    if sum(sizes.values()) != len(ordered):
        raise SplitError(f"sizes sum to {sum(sizes.values())} but there are {len(ordered)} ids")
    out, at = {}, 0
    for name, n in sizes.items():
        out[name] = ordered[at:at + n]
        at += n
    return out


def write_splits(path: str | Path, splits: Mapping[str, Sequence[str]], *,
                 unit: str, rule: str) -> None:
    """Write `splits.json`, recording how the groups were chosen alongside them."""
    validate_splits(splits)
    payload = {"unit": unit, "rule": rule, "splits": {k: list(v) for k, v in splits.items()}}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Read `splits.json`. Give it the file itself or the folder holding it."""
    p = Path(path)
    if p.is_dir():
        p = p / SPLITS_FILENAME
    try:
        payload = json.loads(p.read_text())
    except FileNotFoundError as exc:
        raise SplitError(f"no splits file at {p}. Run `python -m src.data.make_dummy_data` "
                         "to create one, or point `splits:` at a file you already have") from exc
    except json.JSONDecodeError as exc:
        raise SplitError(f"{p}: not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("splits"), dict):
        raise SplitError(f"{p}: expected an object with a 'splits' mapping of "
                         "split name -> list of ids")
    return {k: list(v) for k, v in payload["splits"].items()}


def validate_splits(splits: Mapping[str, Sequence[str]],
                    available: Iterable[str] | None = None) -> None:
    """Raise unless every group has videos, repeats none, and shares none with another."""
    names = list(splits)
    for name in names:
        ids = list(splits[name])
        if not ids:
            raise SplitError(f"split {name!r} is empty")
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise SplitError(f"split {name!r} repeats ids: {dupes}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sorted(set(splits[a]) & set(splits[b]))
            if shared:
                raise SplitError(f"splits {a!r} and {b!r} both contain {shared}. "
                                 "Any score measured this way is really a training score")
    if available is not None:
        known = set(available)
        for name in names:
            missing = sorted(set(splits[name]) - known)
            if missing:
                raise SplitError(f"split {name!r} names videos I could not find: {missing}. "
                                 f"The ones I found were {sorted(known)}")


def resolve_splits(*, available: Sequence[str], splits_file: str = "",
                   explicit: Mapping[str, Sequence[str]] | None = None,
                   data_root: str = "", required: Sequence[str] = ("train", "val"),
                   ) -> dict[str, list[str]]:
    """
    Work out the final groups from the config, then check them.

    A `splits.json` file is read first, and any lists written straight into the config
    override it. I chose not to guess a default, because the obvious guess is putting
    every video in every group, which reports a training score as though it were held out.
    """
    explicit = {k: list(v) for k, v in (explicit or {}).items() if v}
    splits: dict[str, list[str]] = {}
    sources: list[str] = []
    if splits_file:
        path = Path(splits_file)
        if not path.is_absolute() and not path.exists() and data_root:
            path = Path(data_root) / splits_file
        splits.update(load_splits(path))
        sources.append(str(path))
    if explicit:
        splits.update(explicit)
        sources.append("config")
    if not sources:
        raise SplitError(
            "no split configured. Set `splits: <path to splits.json>`, or write out "
            f"{' / '.join(f'{n}_split' for n in required)} lists. I left out any fallback "
            "on purpose, because putting every video in every group reports a training "
            "score as val.")
    source = " overridden by ".join(sources)

    missing = [n for n in required if not splits.get(n)]
    if missing:
        raise SplitError(f"{source}: split(s) {missing} are missing or empty. "
                         f"I found {sorted(splits)}")
    splits = {n: ids for n, ids in splits.items() if ids}
    validate_splits(splits, available)
    return splits


__all__ = ["SPLITS_FILENAME", "SPLIT_PROVENANCE", "SplitError", "assign_splits",
           "write_splits", "load_splits", "validate_splits", "resolve_splits"]
