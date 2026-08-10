"""LoRA injection, trainability marking, and the adapter checkpoint round trip."""

from __future__ import annotations

import pytest
import timm
import torch
import torch.nn as nn

from src.common.lora import (LoRAError, LoRALinear, LoRASpec, adapter_state_dict,
                             apply_lora, load_adapter_state_dict, mark_only_lora_trainable,
                             read_adapter_spec)


class _Block(nn.Module):
    """Layer names match DEFAULT_TARGET_PATTERNS, like SAM2's attention blocks."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.qkv(x))


class _Net(nn.Module):
    """`output_hypernetworks_mlps` is the name SAM2's decoder uses."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.encoder = _Block(d)
        self.output_hypernetworks_mlps = nn.Linear(d, d)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.output_hypernetworks_mlps(self.encoder(x)))


class _Wrapper(nn.Module):
    """Holds the network as `self.model`, like SAM2Segmenter."""

    def __init__(self, net: _Net) -> None:
        super().__init__()
        self.model = net


SPEC = LoRASpec(rank=4, alpha=8, targets=("encoder",))
ALSO_TRAIN = ("output_hypernetworks_mlps", "head")


def build_trained(seed: int = 0) -> tuple[_Net, LoRASpec]:
    """A net with adapters injected and one optimizer step applied."""
    torch.manual_seed(seed)
    net = _Net()
    apply_lora(net, SPEC)
    mark_only_lora_trainable(net, also_train=ALSO_TRAIN)
    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.5)
    loss = ((net(torch.randn(4, 8)) - 1.0) ** 2).mean()
    loss.backward()
    opt.step()
    return net, SPEC


def test_apply_lora_wraps_layers_on_a_real_timm_model():
    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    n = apply_lora(vit, LoRASpec(rank=2, alpha=4, targets=("blocks",)))
    assert n > 0
    assert sum(isinstance(m, LoRALinear) for m in vit.modules()) == n


def test_a_target_that_matches_no_linear_raises():
    net = _Net()
    with pytest.raises(LoRAError, match="0 nn.Linear"):
        apply_lora(net, LoRASpec(rank=2, targets=("head",)))


def test_double_injection_raises():
    net = _Net()
    apply_lora(net, SPEC)
    with pytest.raises(LoRAError):
        apply_lora(net, SPEC)


def test_untrained_adapter_is_the_identity():
    torch.manual_seed(0)
    net = _Net()
    x = torch.randn(3, 8)
    before = net(x)
    apply_lora(net, SPEC)
    assert torch.allclose(net(x), before, atol=1e-6)


def test_adapter_state_dict_saves_every_trainable_tensor():
    """The payload holds every trainable tensor, `also_train` heads included."""
    net, spec = build_trained()
    payload = adapter_state_dict(net, spec, root_name="_Net")
    saved = set(payload["tensors"])
    assert {"head.weight", "head.bias",
            "output_hypernetworks_mlps.weight"} <= saved
    assert saved == {n for n, p in net.named_parameters() if p.requires_grad}


def test_round_trip_restores_lora_and_also_train_weights(tmp_path):
    net, spec = build_trained()
    path = tmp_path / "adapters.pt"
    torch.save(adapter_state_dict(net, spec, root_name="_Net"), path)

    torch.manual_seed(123)  # different init, so a no-op load would be visible
    fresh = _Net()
    apply_lora(fresh, read_adapter_spec(str(path)))
    mark_only_lora_trainable(fresh, also_train=ALSO_TRAIN)
    report = load_adapter_state_dict(fresh, torch.load(path, weights_only=False))

    trained = {n: p for n, p in net.named_parameters() if p.requires_grad}
    assert report.loaded == len(trained) and report.not_in_payload == ()
    for name, param in fresh.named_parameters():
        if param.requires_grad:
            assert torch.equal(param, trained[name]), name


def test_read_adapter_spec_returns_the_rank_that_was_trained(tmp_path):
    net, spec = build_trained()
    path = tmp_path / "adapters.pt"
    torch.save(adapter_state_dict(net, spec, root_name="_Net"), path)
    assert read_adapter_spec(str(path)) == SPEC


def test_loading_into_the_wrong_root_raises_instead_of_dropping_every_key(tmp_path):
    """Saved from the wrapper (keys prefixed `model.`), loaded into the bare net."""
    net, spec = build_trained()
    wrapper = _Wrapper(net)
    payload = adapter_state_dict(wrapper, spec, root_name="_Wrapper")
    assert all(k.startswith("model.") for k in payload["tensors"])

    fresh = _Net()
    apply_lora(fresh, SPEC)
    mark_only_lora_trainable(fresh, also_train=ALSO_TRAIN)
    with pytest.raises(LoRAError, match="none of the"):
        load_adapter_state_dict(fresh, payload)


def test_rank_mismatch_raises_a_named_shape_error():
    net, _ = build_trained()
    payload = adapter_state_dict(net, SPEC, root_name="_Net")

    fresh = _Net()
    apply_lora(fresh, LoRASpec(rank=2, alpha=4, targets=("encoder",)))
    mark_only_lora_trainable(fresh, also_train=ALSO_TRAIN)
    with pytest.raises(LoRAError, match="shape mismatch"):
        load_adapter_state_dict(fresh, payload)


def test_also_train_is_a_path_prefix_not_a_substring():
    """`also_train` entries match on parameter path components, not on substrings."""
    net = _Net()
    apply_lora(net, SPEC)
    before = {n: p.requires_grad for n, p in net.named_parameters()}
    with pytest.raises(LoRAError, match="matched no parameters"):
        mark_only_lora_trainable(net, also_train=("output_hypernetworks",))
    # validation happens before mutation, so a rejected call leaves the model untouched
    assert {n: p.requires_grad for n, p in net.named_parameters()} == before


def test_mark_only_lora_trainable_before_injection_raises():
    with pytest.raises(LoRAError):
        mark_only_lora_trainable(_Net(), also_train=("head",))


def test_wrong_payload_format_raises():
    net = _Net()
    apply_lora(net, SPEC)
    mark_only_lora_trainable(net, also_train=ALSO_TRAIN)
    with pytest.raises(LoRAError):
        load_adapter_state_dict(net, {"format": "torch", "version": 1, "tensors": {}})
