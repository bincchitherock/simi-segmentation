from __future__ import annotations

import pytest

from src.common.config import (ConfigError, PhaseConfig, SegConfig,
                               config_from_mapping, config_to_dict, load_config)


def write_yaml(path, body: str):
    path.write_text(body)
    return path


def test_unknown_key_raises_and_names_the_key(tmp_path):
    cfg = write_yaml(tmp_path / "c.yaml", "data_root: d\nimg_sz: 224\n")
    with pytest.raises(ConfigError) as exc:
        load_config(cfg, PhaseConfig)
    assert "img_sz" in str(exc.value)
    assert "img_size" in str(exc.value)  # suggestion
    assert str(cfg) in str(exc.value)


def test_missing_required_key_raises_with_the_key_name(tmp_path):
    cfg = write_yaml(tmp_path / "c.yaml", "epochs: 3\n")
    with pytest.raises(ConfigError) as exc:
        load_config(cfg, PhaseConfig)
    assert "data_root" in str(exc.value)


def test_yaml_1e4_float_form_is_rejected_with_an_explanation(tmp_path):
    cfg = write_yaml(tmp_path / "c.yaml", "data_root: d\nlr: 1e-4\n")
    with pytest.raises(ConfigError) as exc:
        load_config(cfg, PhaseConfig)
    assert "1.0e-4" in str(exc.value)


def test_overrides_win_but_none_entries_are_dropped(tmp_path):
    cfg = write_yaml(tmp_path / "c.yaml", "data_root: d\nepochs: 7\nseed: 3\n")
    loaded = load_config(cfg, PhaseConfig, overrides={"epochs": 1, "seed": None})
    assert (loaded.epochs, loaded.seed) == (1, 3)


def test_config_round_trips_so_a_checkpoint_records_the_run_it_came_from(tmp_path):
    cfg = load_config(write_yaml(tmp_path / "c.yaml",
                                 "data_root: d\ninstrument_loss_weight: 50.0\ndropout: 0.9\n"),
                      PhaseConfig)
    assert config_from_mapping(config_to_dict(cfg), PhaseConfig) == cfg
    assert cfg.instrument_loss_weight == 50.0 and cfg.dropout == 0.9


def test_stride_larger_than_clip_len_is_rejected(tmp_path):
    cfg = write_yaml(tmp_path / "c.yaml", "data_root: d\nclip_len: 16\nstride: 32\n")
    with pytest.raises(ConfigError):
        load_config(cfg, PhaseConfig)


def test_sam2_model_without_checkpoint_raises_instead_of_downgrading(tmp_path):
    cfg = write_yaml(tmp_path / "s.yaml", "data_root: d\nmodel: sam2\n")
    with pytest.raises(ConfigError) as exc:
        load_config(cfg, SegConfig)
    assert "checkpoint" in str(exc.value) and "model_cfg" in str(exc.value)


def test_tinyunet_is_the_explicit_opt_in(tmp_path):
    cfg = load_config(write_yaml(tmp_path / "s.yaml", "data_root: d\nmodel: tinyunet\n"),
                      SegConfig)
    assert cfg.model == "tinyunet"


def test_shipped_configs_load(tmp_path):
    """Every config in configs/ satisfies its own schema."""
    from pathlib import Path

    seg_names = {"seg_sam2_lora.yaml", "seg_phantom.yaml"}
    for path in sorted(Path("configs").glob("*.yaml")):
        schema = SegConfig if path.name in seg_names else PhaseConfig
        load_config(path, schema)
