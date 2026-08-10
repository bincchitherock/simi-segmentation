from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from conftest import REPO_ROOT, run_module, write_phase_videos
from src.common.config import PhaseConfig, SegConfig, load_config
from src.data.splits import SplitError, resolve_splits

CONFIG_DIR = REPO_ROOT / "configs"
SEG_CONFIGS = {"seg_sam2_lora.yaml", "seg_phantom.yaml"}


def read_splits_file(path: Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text())
    assert isinstance(payload.get("splits"), dict), f"{path}: no 'splits' mapping"
    assert payload.get("unit") == "video_id", f"{path}: splits must be video-level"
    return payload["splits"]


# Split helper

def test_no_split_configured_raises():
    with pytest.raises(SplitError, match="no split configured"):
        resolve_splits(available=["a", "b"])


def test_overlapping_splits_raise():
    with pytest.raises(SplitError, match="overlap"):
        resolve_splits(available=["a", "b"], explicit={"train": ["a", "b"], "val": ["b"]})


def test_empty_split_raises():
    with pytest.raises(SplitError):
        resolve_splits(available=["a", "b"], explicit={"train": ["a"], "val": []})


def test_split_naming_an_unknown_video_raises():
    with pytest.raises(SplitError, match="unknown"):
        resolve_splits(available=["a"], explicit={"train": ["a"], "val": ["ghost"]})


def test_explicit_lists_win_over_the_splits_file(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps(
        {"unit": "video_id", "rule": "test", "splits": {"train": ["a"], "val": ["b"]}}))
    got = resolve_splits(available=["a", "b", "c"], data_root=str(tmp_path),
                         splits_file="splits.json",
                         explicit={"train": ["c"], "val": ["a"]})
    assert got == {"train": ["c"], "val": ["a"]}


def test_explicit_lists_override_per_split_and_keep_the_rest_of_the_file(tmp_path):
    """An explicit list replaces only its own split; the file's other splits survive."""
    (tmp_path / "splits.json").write_text(json.dumps(
        {"unit": "video_id", "rule": "test",
         "splits": {"train": ["a"], "val": ["b"], "test": ["d"]}}))
    got = resolve_splits(available=["a", "b", "c", "d"], data_root=str(tmp_path),
                         splits_file="splits.json",
                         explicit={"train": ["c"], "val": ["a"]})
    assert got == {"train": ["c"], "val": ["a"], "test": ["d"]}


# Shipped artifacts

@pytest.mark.parametrize("name", ["phase_spatiotemporal.yaml", "phase_phantom.yaml"])
def test_shipped_phase_configs_declare_a_split(name):
    cfg = load_config(CONFIG_DIR / name, PhaseConfig)
    assert cfg.splits or (cfg.train_split and cfg.val_split), \
        f"{name} declares no split; the trainer must not be able to guess"
    assert not set(cfg.train_split) & set(cfg.val_split)


@pytest.mark.parametrize("name", sorted(SEG_CONFIGS))
def test_shipped_seg_configs_declare_a_split(name):
    cfg = load_config(CONFIG_DIR / name, SegConfig)
    assert cfg.splits or (cfg.train_split and cfg.val_split), \
        f"{name} declares no split; `val = train` is how the audit found it"


def test_generated_splits_are_non_empty_and_disjoint(generated_data):
    found = sorted(generated_data.rglob("splits.json"))
    assert found, f"make_dummy_data wrote no splits.json under {generated_data}"
    for path in found:
        splits = read_splits_file(path)
        assert {"train", "val"} <= set(splits), f"{path}: {sorted(splits)}"
        for a in splits:
            assert splits[a], f"{path}: split {a!r} is empty"
            for b in splits:
                if a != b:
                    assert not set(splits[a]) & set(splits[b]), f"{path}: {a}/{b} overlap"


def test_generated_split_ids_all_exist_in_the_manifest(generated_data):
    manifest = json.loads((generated_data / "phase" / "videos.json").read_text())
    known = {v["video_id"] for v in manifest}
    splits = read_splits_file(generated_data / "phase" / "splits.json")
    ids = {i for v in splits.values() for i in v}
    assert ids <= known, f"unknown video ids {sorted(ids - known)}"
    assert ids == known, f"videos in no split: {sorted(known - ids)}"


# Trainer should refuse ambiguous split.

def tiny_phase_config(tmp_path: Path, **extra) -> Path:
    root = write_phase_videos(tmp_path / "phase", {"video_00": 8, "video_01": 8},
                              size=32, num_steps=4, num_instruments=3)
    body = {"data_root": str(root), "backbone": "timm:convnext_atto", "pretrained": False,
            "temporal_ch": 32, "temporal_layers": 2, "n_heads": 4,
            "num_steps": 4, "num_instruments": 3, "clip_len": 8, "stride": 8,
            "img_size": 32, "batch_size": 2, "workers": 0, "epochs": 1, "seed": 0}
    body.update(extra)
    path = tmp_path / "phase.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def train_with(cfg: Path, out: Path):
    return run_module("src.phase.train", "--config", str(cfg), "--out", str(out))


def test_training_refuses_a_config_with_no_split(tmp_path):
    proc = train_with(tiny_phase_config(tmp_path), tmp_path / "run")
    assert proc.returncode != 0, "a config with no split must not train"
    assert "split" in (proc.stdout + proc.stderr).lower()


def test_training_refuses_overlapping_train_and_val_splits(tmp_path):
    cfg = tiny_phase_config(tmp_path, train_split=["video_00", "video_01"],
                            val_split=["video_01"])
    proc = train_with(cfg, tmp_path / "run")
    assert proc.returncode != 0, "overlapping splits must not train"
    assert "split" in (proc.stdout + proc.stderr).lower()


def test_training_refuses_an_empty_split(tmp_path):
    cfg = tiny_phase_config(tmp_path, train_split=["video_00"], val_split=[])
    proc = train_with(cfg, tmp_path / "run")
    assert proc.returncode != 0, "a split that selects no video must not train"


@pytest.mark.slow
def test_training_runs_end_to_end_on_the_real_dataset_classes(tmp_path):
    cfg = tiny_phase_config(tmp_path, train_split=["video_00"], val_split=["video_01"])
    out = tmp_path / "run"
    proc = train_with(cfg, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "best.pt").exists()


@pytest.mark.slow
def test_training_reads_a_splits_json(tmp_path):
    cfg_path = tiny_phase_config(tmp_path, splits="splits.json")
    data_root = Path(yaml.safe_load(cfg_path.read_text())["data_root"])
    (data_root / "splits.json").write_text(json.dumps(
        {"unit": "video_id", "rule": "test fixture",
         "splits": {"train": ["video_00"], "val": ["video_01"]}}))
    proc = train_with(cfg_path, tmp_path / "run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
