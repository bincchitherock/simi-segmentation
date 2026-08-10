"""Segmentation prompts, the toy segmenter, and the SAM2 import guard."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.common.lora import LoRASpec
from src.common.seed import make_generator
from src.data.splits import SplitError
from src.segmentation.dataset import MaskDataset, prompt_from_mask, resolve_seg_splits
from src.segmentation.sam2_lora import SAM2Segmenter, TinyUNetSegmenter

SPEC = LoRASpec(rank=4, alpha=8, targets=("",))


@pytest.fixture(scope="module")
def seg_root(generated_data) -> str:
    return str(generated_data / "seg")


def make_dataset(root: str, split: tuple[str, ...], **kwargs) -> MaskDataset:
    return MaskDataset(root, split, img_size=64, seed=0, **kwargs)

# Prompts

def test_point_prompt_is_inside_the_mask_and_labelled_foreground():
    mask = torch.zeros(16, 16)
    mask[4:8, 6:10] = 1.0
    coords, labels, box = prompt_from_mask(mask, "point", make_generator(0))
    assert box is None
    x, y = coords.reshape(2).tolist()
    assert mask[int(y), int(x)] > 0.5
    assert labels.reshape(-1).tolist() == [1]


def test_the_same_generator_seed_gives_the_same_point():
    mask = torch.zeros(32, 32)
    mask[2:30, 2:30] = 1.0
    torch.manual_seed(7)   # the global stream must be irrelevant
    a = prompt_from_mask(mask, "point", make_generator(5))[0]
    torch.manual_seed(99)
    b = prompt_from_mask(mask, "point", make_generator(5))[0]
    assert torch.equal(a, b)


def test_box_prompt_is_the_tight_bounding_box_of_the_mask():
    mask = torch.zeros(16, 16)
    mask[4:9, 6:11] = 1.0
    coords, labels, box = prompt_from_mask(mask, "box", make_generator(0))
    assert coords is None and labels is None
    assert box.tolist() == [6.0, 4.0, 10.0, 8.0]


@pytest.mark.parametrize("kind", ["point", "box"])
def test_an_empty_mask_cannot_be_prompted(kind):
    """An empty mask has nothing to point at, and a stand-in prompt would leak the answer."""
    with pytest.raises(ValueError, match="mask is empty"):
        prompt_from_mask(torch.zeros(16, 16), kind, make_generator(0))


def test_unknown_prompt_kind_raises():
    with pytest.raises(ValueError):
        prompt_from_mask(torch.zeros(8, 8), "scribble", make_generator(0))


# Dataset: splits and reproducibility

def test_train_and_val_are_different_pairs(seg_root):
    splits = resolve_seg_splits_from_disk(seg_root)
    train = make_dataset(seg_root, splits["train"], name="train")
    val = make_dataset(seg_root, splits["val"], name="val")
    train_ids = {p["pair_id"] for p in (train[i] for i in range(len(train)))}
    val_ids = {p["pair_id"] for p in (val[i] for i in range(len(val)))}
    assert train_ids and val_ids and not train_ids & val_ids


def resolve_seg_splits_from_disk(root: str) -> dict[str, tuple[str, ...]]:
    from src.common.config import SegConfig

    return resolve_seg_splits(SegConfig(data_root=root, model="tinyunet",
                                        splits="splits.json"))


def test_pairs_with_an_empty_mask_are_dropped_and_reported(seg_root, capsys):
    """A frame with no instrument is not a prompted-segmentation sample, and the count
    dropped is printed.
    """
    from src.segmentation.dataset import read_pairs

    splits = resolve_seg_splits_from_disk(seg_root)
    ds = make_dataset(seg_root, splits["train"], name="train")
    on_disk = [p for p in read_pairs(seg_root) if p["video_id"] in set(splits["train"])]

    assert all(ds[i]["mask"].max() > 0.5 for i in range(len(ds)))
    assert len(ds) < len(on_disk), "the phantom must contain some empty masks to drop"
    assert f"{len(on_disk) - len(ds)} of {len(on_disk)} dropped" in capsys.readouterr().out


def test_a_split_naming_no_known_video_raises(seg_root):
    with pytest.raises(SplitError):
        make_dataset(seg_root, ("no_such_clip",))


def test_prompts_are_identical_across_dataloader_worker_counts(seg_root):
    splits = resolve_seg_splits_from_disk(seg_root)
    ds = make_dataset(seg_root, splits["val"], name="val")
    single = torch.cat([b["point_coords"] for b in DataLoader(ds, batch_size=2, num_workers=0)])
    multi = torch.cat([b["point_coords"] for b in DataLoader(ds, batch_size=2, num_workers=2)])
    assert torch.equal(single, multi)


def test_prompts_do_not_move_between_epochs(seg_root):
    splits = resolve_seg_splits_from_disk(seg_root)
    ds = make_dataset(seg_root, splits["val"], name="val")
    first = [ds[i]["point_coords"] for i in range(len(ds))]
    torch.manual_seed(4321)
    assert all(torch.equal(a, ds[i]["point_coords"]) for i, a in enumerate(first))


# The toy segmenter

def test_tiny_unet_actually_reads_the_prompt():
    torch.manual_seed(0)
    model = TinyUNetSegmenter(SPEC).eval()
    images = torch.randn(1, 3, 32, 32)
    left = model(images, point_coords=torch.tensor([[[4.0, 4.0]]]),
                 point_labels=torch.tensor([[1]]))
    right = model(images, point_coords=torch.tensor([[[28.0, 28.0]]]),
                  point_labels=torch.tensor([[1]]))
    assert not torch.equal(left, right)


def test_tiny_unet_requires_a_prompt():
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        TinyUNetSegmenter(SPEC).eval()(torch.randn(1, 3, 32, 32))


@pytest.mark.parametrize("img_size", [32, 33])
def test_tiny_unet_handles_odd_input_sizes(img_size):
    torch.manual_seed(0)
    model = TinyUNetSegmenter(SPEC).eval()
    out = model(torch.randn(1, 3, img_size, img_size),
                point_coords=torch.tensor([[[1.0, 1.0]]]),
                point_labels=torch.tensor([[1]]))
    assert out.shape == (1, 1, img_size, img_size)


@pytest.mark.parametrize("base", [4, 12, 16])
def test_tiny_unet_accepts_channel_counts_that_are_not_multiples_of_eight(base):
    torch.manual_seed(0)
    TinyUNetSegmenter(SPEC, base=base)


@pytest.mark.parametrize("train_decoder", [True, False])
def test_train_decoder_is_live_on_the_tinyunet_backend_too(train_decoder):
    torch.manual_seed(0)
    model = TinyUNetSegmenter(SPEC, train_decoder=train_decoder)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora_" in n for n in trainable), "the adapters always train"
    assert any(n.startswith(("dec.", "head.")) for n in trainable) is train_decoder


def test_tiny_unet_names_itself_honestly():
    torch.manual_seed(0)
    assert TinyUNetSegmenter(SPEC).backend == "tinyunet"


def test_tiny_unet_init_depends_on_its_seed_and_not_on_the_ambient_rng():
    """TinyUNet's frozen weights are random rather than loaded, so an adapter file only
    reproduces a run if `init_seed` alone determines them.
    """
    torch.manual_seed(0)
    a = TinyUNetSegmenter(SPEC, init_seed=3)
    torch.randn(1000)
    b = TinyUNetSegmenter(SPEC, init_seed=3)
    c = TinyUNetSegmenter(SPEC, init_seed=4)

    frozen = [n for n, p in a.named_parameters() if not p.requires_grad]
    assert frozen, "the base weights must be frozen, or there is no LoRA here"
    for name in frozen:
        assert torch.equal(a.get_parameter(name), b.get_parameter(name)), name
    assert any(not torch.equal(a.get_parameter(n), c.get_parameter(n)) for n in frozen), \
        "a different seed must still give a different init"

# SAM2: not installed here, so only the import guard is exercised 

def test_sam2_segmenter_raises_an_informative_import_error():
    with pytest.raises(ImportError, match="sam2"):
        SAM2Segmenter(checkpoint="nonexistent.pt",
                      model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", spec=SPEC)
