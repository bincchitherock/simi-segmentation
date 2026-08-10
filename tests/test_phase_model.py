"""SpatioTemporalMultiTask: shapes, temporal-stack sizing, and padding masking.

Padded frames are duplicates of the last real frame; they must reach neither the loss nor
the predictions of real frames.
"""

from __future__ import annotations

import pytest
import torch

from src.common.config import PhaseConfig
from src.phase.model import PhaseModelConfig, SpatioTemporalMultiTask

B, T, IMG = 2, 8, 32
NUM_STEPS, NUM_INSTRUMENTS, N_VALID = 5, 4, 5

COMMON = dict(backbone="timm:convnext_atto", pretrained=False, temporal_ch=32,
              temporal_layers=3, n_heads=4, dropout=0.0, num_steps=NUM_STEPS,
              num_instruments=NUM_INSTRUMENTS, clip_len=T)


def build(**overrides) -> SpatioTemporalMultiTask:
    torch.manual_seed(0)
    return SpatioTemporalMultiTask(PhaseModelConfig(**{**COMMON, **overrides})).eval()


def valid_mask() -> torch.Tensor:
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :N_VALID] = True
    return mask


def targets() -> tuple[torch.Tensor, torch.Tensor]:
    step = torch.full((B, T), -100, dtype=torch.long)
    step[:, :N_VALID] = torch.randint(0, NUM_STEPS, (B, N_VALID))
    instr = torch.zeros(B, T, NUM_INSTRUMENTS)
    instr[:, :N_VALID] = (torch.rand(B, N_VALID, NUM_INSTRUMENTS) > 0.5).float()
    return step, instr

# Shapes and construction
@pytest.mark.parametrize("temporal", ["tcn", "transformer"])
def test_forward_returns_the_documented_shapes(temporal):
    out = build(temporal=temporal)(torch.randn(B, T, 3, IMG, IMG))
    assert out["step"].shape == (B, T, NUM_STEPS)
    assert out["instrument"].shape == (B, T, NUM_INSTRUMENTS)


def test_temporal_stack_is_sized_from_the_encoder_not_from_temporal_ch():
    model = build(temporal="tcn", temporal_ch=32)
    assert model.encoder.feat_dim != 32          # convnext_atto is 320-d
    assert model.temporal.in_proj.in_channels == model.encoder.feat_dim


def test_tcn_dilations_never_exceed_the_clip_length():
    model = build(temporal="tcn", temporal_layers=8)
    assert max(layer.conv.dilation[0] for layer in model.temporal.layers) <= T


def test_model_config_copies_every_shared_field_from_the_run_config():
    cfg = PhaseConfig(data_root="d", instrument_loss_weight=7.5, dropout=0.42,
                      temporal_ch=32, num_steps=NUM_STEPS,
                      num_instruments=NUM_INSTRUMENTS, clip_len=T, stride=T)
    model_cfg = PhaseModelConfig.from_config(cfg)
    assert model_cfg.instrument_loss_weight == 7.5
    assert model_cfg.dropout == 0.42
    for field in ("backbone", "temporal", "temporal_ch", "temporal_layers", "n_heads",
                  "num_steps", "num_instruments", "clip_len"):
        assert getattr(model_cfg, field) == getattr(cfg, field), field


# Padding must not be data
def test_padded_frames_contribute_nothing_to_the_instrument_loss():
    torch.manual_seed(0)
    model = build()
    out = {"step": torch.randn(B, T, NUM_STEPS),
           "instrument": torch.randn(B, T, NUM_INSTRUMENTS)}
    step, instr = targets()
    flipped = instr.clone()
    flipped[:, N_VALID:] = 1.0

    total_a, parts_a = model.compute_loss(out, step, instr, valid_mask())
    total_b, parts_b = model.compute_loss(out, step, flipped, valid_mask())
    assert torch.allclose(total_a, total_b), "padded instrument targets leaked into the loss"
    assert parts_a["instrument"] == pytest.approx(parts_b["instrument"])


def test_instrument_loss_equals_the_loss_over_valid_frames_alone():
    torch.manual_seed(0)
    model = build()
    out = {"step": torch.randn(B, T, NUM_STEPS),
           "instrument": torch.randn(B, T, NUM_INSTRUMENTS)}
    step, instr = targets()

    _, parts = model.compute_loss(out, step, instr, valid_mask())
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        out["instrument"][:, :N_VALID], instr[:, :N_VALID])
    assert parts["instrument"] == pytest.approx(expected.item(), rel=1e-5)


def test_instrument_loss_weight_scales_the_total():
    torch.manual_seed(0)
    out = {"step": torch.randn(B, T, NUM_STEPS),
           "instrument": torch.randn(B, T, NUM_INSTRUMENTS)}
    step, instr = targets()
    one, parts = build(instrument_loss_weight=1.0).compute_loss(out, step, instr, valid_mask())
    ten, _ = build(instrument_loss_weight=10.0).compute_loss(out, step, instr, valid_mask())
    assert ten.item() == pytest.approx(one.item() + 9 * parts["instrument"], rel=1e-5)


def test_a_batch_with_no_valid_frame_raises_instead_of_returning_nan():
    torch.manual_seed(0)
    model = build()
    out = {"step": torch.randn(B, T, NUM_STEPS),
           "instrument": torch.randn(B, T, NUM_INSTRUMENTS)}
    step, instr = targets()
    with pytest.raises(ValueError):
        model.compute_loss(out, step, instr, torch.zeros(B, T, dtype=torch.bool))


@pytest.mark.parametrize("temporal", ["tcn", "transformer"])
def test_padded_timesteps_do_not_change_the_predictions_of_real_frames(temporal):
    torch.manual_seed(0)
    model = build(temporal=temporal)
    clip = torch.randn(B, T, 3, IMG, IMG)
    valid = valid_mask()

    before = model(clip, valid)["step"][:, :N_VALID]
    perturbed = clip.clone()
    perturbed[:, N_VALID:] = torch.randn(B, T - N_VALID, 3, IMG, IMG) * 5.0
    after = model(perturbed, valid)["step"][:, :N_VALID]
    assert torch.allclose(before, after, atol=1e-5)


def test_padding_cannot_reach_batchnorm_statistics_in_train_mode():
    torch.manual_seed(0)
    model = SpatioTemporalMultiTask(
        PhaseModelConfig(**{**COMMON, "backbone": "timm:resnet18"})).train()
    clip = torch.randn(B, T, 3, IMG, IMG)
    valid = valid_mask()

    before = model(clip, valid)["step"][:, :N_VALID]
    perturbed = clip.clone()
    perturbed[:, N_VALID:] = torch.randn(B, T - N_VALID, 3, IMG, IMG) * 5.0
    after = model(perturbed, valid)["step"][:, :N_VALID]
    assert torch.allclose(before, after, atol=1e-5)
