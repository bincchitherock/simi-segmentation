# results/

Everything here came off an Apple M5 CPU and took about four minutes for me.

```bash
PSAI_FORCE_CPU=1 bash scripts/reproduce_results.sh
```

## How to read a split

Three groups of videos, each with a different job.

- **train** is what the model learns from. 
- **val** picks which saved copy of the model to keep.
- **test** is looked at once, at the very end. lowkey is the only number worth quoting.

Videos never appear in more than one group. `src.data.splits.resolve_splits` raises an error
if they do.

## Instrument segmentation

TinyUNet, 1422 training updates, seed 0. Splits are in `data/seg/splits.json`.

| split | clips | masks scored | dropped as empty | used for |
|---|---|---|---|---|
| train | `clip_00` to `clip_05` | 71 | 1 | learning |
| val | `clip_06`, `clip_07` | 20 | 4 | choosing which epoch to keep |
| test | `clip_08`, `clip_09` | 24 | 0 | scored once, never seen before |

| | val (chose the model) | **test (held out from both)** |
|---|---|---|
| Dice | 0.964 ± 0.007 | **0.959 ± 0.010** |
| IoU | 0.931 ± 0.013 | **0.922 ± 0.018** |
| same model **untrained**, Dice | 0.079 ± 0.023 | **0.121 ± 0.028** |
| same model **untrained**, IoU | 0.041 ± 0.012 | **0.065 ± 0.016** |

## Surgical phase and instrument recognition

ResNet18 trained from scratch, followed by a temporal model. Seed 0. Splits are in
`data/phase/splits.json`, 6 train / 3 val / 3 test. Epoch 11 was kept, chosen on a val step
macro-F1 of 0.249.

Scored on the test videos `video_09`, `video_10` and `video_11`, 144 frames, held out from
both the training and the choice of epoch:

| | step acc | step macro-F1 (14 steps) | edit | F1@10 | F1@25 | F1@50 | instrument macro-F1 |
|---|---|---|---|---|---|---|---|
| **trained** | **0.340** | **0.203** | **17.8** | **25.4** | **19.0** | **12.7** | **0.221** |
| pure guessing | 0.069 | 0.052 | 8.2 | 4.0 | 1.3 | 0.0 | 0.123 |
| same network, untrained | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.074 |

It's two floors because they fail differently. Guessing at random spreads its answers across all
14 steps and picks up a little credit by luck. The untrained network puts everything in one
step and scores zero, but obviously a real result has to beat both.

The step head clears its floor by roughly five times. The instrument head, 0.221 against
0.123, does not clear it by enough to call it working. That is the honest read of this run.

## Files

| file | what it is |
|---|---|
| `segmentation_metrics.json` | val scores, per-mask, plus the settings that produced them |
| `segmentation_test_metrics.json` | the same on the held-out test clips |
| `segmentation_qualitative.png` | val masks: input, truth, prediction, both outlines together |
| `segmentation_test_qualitative.png` | the same on test |
| `segmentation_untrained_baseline.png` | the same masks before any training |
| `segmentation_test_untrained_baseline.png` | the same on test |
| `segmentation_training_curve.png` | training loss, and val score per epoch |
| `phase_metrics.json` | phase scores, both floors, and per-video accuracy |
| `phase_timeline.png` | true steps above predicted steps, as coloured strips over time |
| `*.log` | console output from the run that produced all of the above |

In the pictures, masks are always chosen spread evenly from worst to best, with the
worst at the top.
