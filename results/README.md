# results/

I produced everything here on an Apple M5 CPU.
Took about 4 minutes.

```bash
PSAI_FORCE_CPU=1 bash scripts/reproduce_results.sh
```

## Headline results

### Instrument segmentation: TinyUNet, 1422 optimizer steps, seed 0

Splits are by clip and disjoint by construction (`data/seg/splits.json`,
validated by `src.data.splits.resolve_splits`, which raises on any overlap).

| split | clips | masks scored | dropped (empty) | used for |
|---|---|---|---|---|
| train | `clip_00` … `clip_05` | 71 | 1 | gradient updates |
| val | `clip_06`, `clip_07` | 20 | 4 | checkpoint selection only |
| test | `clip_08`, `clip_09` | 24 | 0 | scored once, never seen |

| | val (selected the checkpoint) | **test (held out from both)** |
|---|---|---|
| Dice | 0.964 ± 0.007 | **0.959 ± 0.010** |
| IoU | 0.931 ± 0.013 | **0.922 ± 0.018** |
| same model **untrained**, Dice | 0.079 ± 0.023 | **0.121 ± 0.028** |
| same model **untrained**, IoU | 0.041 ± 0.012 | **0.065 ± 0.016** |

### Surgical phase + instrument recognition — ResNet18(scratch)+TCN, seed 0

Splits are by video (`data/phase/splits.json`): 6 train / 3 val / 3 test,
disjoint. Checkpoint selected at epoch 11 by val step macro-F1 = 0.249.

Scored on the **test** videos (`video_09`, `video_10`, `video_11`; 144 frames),
held out from both training and selection:

| | step acc | step macro-F1 (14-way) | edit | F1@10 | F1@25 | F1@50 | instr macro-F1 |
|---|---|---|---|---|---|---|---|
| **trained** | **0.340** | **0.203** | **17.8** | **25.4** | **19.0** | **12.7** | **0.221** |
| uniform-random logits | 0.069 | 0.052 | 8.2 | 4.0 | 1.3 | 0.0 | 0.123 |
| same network, untrained | 0.000 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 | 0.074 |