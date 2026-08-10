"""Segmentation train -> save -> reload -> score, end to end on the phantom."""

from __future__ import annotations

import json

import pytest
import yaml

from conftest import run_module

pytestmark = pytest.mark.slow


def seg_config(tmp_path, seg_root) -> str:
    body = {"data_root": str(seg_root), "model": "tinyunet", "img_size": 64,
            "prompt_kind": "point", "splits": "splits.json", "lora_rank": 4,
            "lora_alpha": 8, "lr": 1.0e-3, "epochs": 2, "batch_size": 4,
            "workers": 0, "seed": 0}
    path = tmp_path / "seg.yaml"
    path.write_text(yaml.safe_dump(body))
    return str(path)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, generated_data) -> tuple[str, str]:
    """(config path, run dir) for one short TinyUNet training run."""
    tmp = tmp_path_factory.mktemp("seg_run")
    cfg = seg_config(tmp, generated_data / "seg")
    proc = run_module("src.segmentation.train_sam2", "--config", cfg, "--out", str(tmp / "run"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return cfg, str(tmp / "run")


def test_training_writes_an_adapter_checkpoint(trained_run):
    import torch

    cfg, run_dir = trained_run
    payload = torch.load(f"{run_dir}/adapters.pt", map_location="cpu", weights_only=False)
    assert payload["format"] == "psai-lora"
    assert payload["backend"] == "tinyunet"
    # the `also_train` heads belong in the file too, not just the LoRA A/B matrices
    saved = set(payload["tensors"])
    assert any(k.startswith("head.") for k in saved), sorted(saved)
    assert any(k.startswith("dec.") for k in saved), sorted(saved)
    assert any("lora_A" in k for k in saved), sorted(saved)


def test_reloading_the_checkpoint_reproduces_the_val_dice(trained_run, tmp_path):
    """Scoring the reloaded checkpoint gives the Dice the run recorded."""
    import torch

    cfg, run_dir = trained_run
    out = tmp_path / "results"
    proc = run_module("src.segmentation.predict", "--checkpoint", f"{run_dir}/adapters.pt",
                      "--config", cfg, "--split", "val", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = torch.load(f"{run_dir}/adapters.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["dice"]["mean"] == pytest.approx(payload["val_dice"], abs=1e-4)


def test_reported_metrics_name_their_population_and_their_backend(trained_run, tmp_path):
    """metrics.json records the backend, the split, and the scored population."""
    cfg, run_dir = trained_run
    out = tmp_path / "results"
    proc = run_module("src.segmentation.predict", "--checkpoint", f"{run_dir}/adapters.pt",
                      "--config", cfg, "--split", "val", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["backend"] == "tinyunet", "a TinyUNet score must never read as SAM2"
    assert metrics["split"] == "val" and metrics["split_video_ids"]
    assert metrics["n_scored"] + metrics["n_excluded_empty_pairs"] == metrics["n_samples"]
    assert (out / "qualitative.png").exists()


def test_a_config_that_would_build_a_different_model_is_refused(trained_run, tmp_path,
                                                                generated_data):
    cfg, run_dir = trained_run
    body = yaml.safe_load(open(cfg))
    body["seed"] = body["seed"] + 1
    other = tmp_path / "other_seed.yaml"
    other.write_text(yaml.safe_dump(body))

    proc = run_module("src.segmentation.predict", "--checkpoint", f"{run_dir}/adapters.pt",
                      "--config", str(other), "--split", "val", "--out", str(tmp_path / "r"))
    assert proc.returncode != 0
    assert "would build a different model" in proc.stdout + proc.stderr


def test_the_val_split_is_not_the_train_split(trained_run, generated_data):
    import torch

    from src.data.splits import load_splits

    _, run_dir = trained_run
    payload = torch.load(f"{run_dir}/adapters.pt", map_location="cpu", weights_only=False)
    splits = load_splits(generated_data / "seg")
    assert payload["val_split"] == splits["val"], "the checkpoint must record the split it used"
    assert splits["val"], "the val split must name at least one clip"
    assert not set(splits["train"]) & set(payload["val_split"])


def test_the_test_split_is_held_out_from_training_and_from_selection(trained_run, tmp_path,
                                                                     generated_data):
    import torch

    from src.data.splits import load_splits

    cfg, run_dir = trained_run
    out = tmp_path / "test_results"
    proc = run_module("src.segmentation.predict", "--checkpoint", f"{run_dir}/adapters.pt",
                      "--config", cfg, "--split", "test", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    payload = torch.load(f"{run_dir}/adapters.pt", map_location="cpu", weights_only=False)
    splits = load_splits(generated_data / "seg")
    metrics = json.loads((out / "metrics.json").read_text())
    scored = set(metrics["split_video_ids"])

    assert scored == set(splits["test"]) and scored
    assert not scored & set(splits["train"]), "test overlaps the fitted clips"
    assert not scored & set(payload["val_split"]), "test overlaps the selection clips"
    assert "held out" in metrics["split_provenance"]
