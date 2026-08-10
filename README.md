# Pituitary Surgical Segmentation

- Author: Seo Bin Han. 
- Endoscopic pituitary surgery segmentation techniques explored during the 2024 SIMI Lab internship.

> NOTE: Most of the work here is independent curiosity + research following the literature
> review done at SIMI, so the data and progress available to those on the project are not
> incorporated here.

The procedure is performed through the nose with an endoscope, so the video stream is the
only continuous record of what happened. Two prototypes follow from that, as separate models
over the same video:

1. **Instrument segmentation**: pixel-accurate outlines, the substrate for tool-tissue
   distance, motion economy, and any future AR overlay.
2. **Workflow recognition**: which step is happening and which instruments are in view, by
   frame. Pituitary surgery suits this: few steps, near-fixed order, narrow corridor.

The intended loop is that segmentation feeds workflow. They share the data contract, split
logic, seeding, config loader and metric conventions in [`src/common/`](src/common) and
[`src/data/`](src/data).

## Data + Metrics

Every metric here was measured on [`src/data/phantom.py`](src/data/phantom.py), which I
procedurally generated as an endoscopic phantom. It is not surgical data, and no number here
is a clinical or benchmark result. The mask is rendered from the same geometry as the
instruments rather than added into the pixels, so the best Dice any single global intensity
threshold reaches is mean 0.28, which means that a model has to learn shape and appearance.

**SAM 2 is not installed here and is never executed.**
[`configs/seg_phantom.yaml`](configs/seg_phantom.yaml) trains `TinyUNetSegmenter`, a
~30K-parameter U-Net, so its Dice is a TinyUNet number. The SAM 2 code paths were written
against `facebookresearch/sam2 @ main` and print `UNVERIFIED PATH` at runtime.

## Results on the phantom, seed 0

10 clips split 6 / 2 / 2 by clip id, 1422 optimizer steps, `val` selected epoch 78:

| 24 masks / 2 held-out test clips | trained | same model untrained |
|---|---|---|
| Dice | **0.959 ± 0.010** | 0.121 ± 0.028 |
| IoU | **0.922 ± 0.018** | 0.065 ± 0.016 |

Both columns are oracle-prompted. I.e., each mask is scored after the model is handed a positive
click sampled from that mask's own ground truth (the standard SAM protocol). This measures
outline quality given a click, not instrument detection. Frames with no instrument in view
are dropped rather than scored, as there is nothing to click.

![Held-out test predictions: input, ground truth, prediction, and the two contours overlaid](results/segmentation_test_qualitative.png)

Phase recognition, 12 videos split 6 / 3 / 3, scored once on 144 held-out frames: step
accuracy 0.340 and macro-F1 0.203 against measured floors of 0.069 / 0.052, segmental edit
17.8 against 8.2. The step head is ~4-5x its floor; the **instrument head, at 0.221 against
a 0.123 floor, is the weak part of this run** and should not be described as working.

Every artifact explained, with the caveats: [`results/README.md`](results/README.md).

## Quickstart

Python 3.10+. `PSAI_FORCE_CPU=1` forces CPU and makes runs bit-exact. MPS has no deterministic-algorithms switch, so use it for anything you
intend to compare.

```bash
pip install -r requirements.txt

python -m src.data.make_dummy_data --out data
PSAI_FORCE_CPU=1 bash scripts/smoke_test.sh
PSAI_FORCE_CPU=1 bash scripts/reproduce_results.sh
PSAI_FORCE_CPU=1 python -m pytest
```
