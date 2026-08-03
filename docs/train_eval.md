# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

## Two-phase training (original workflow)

Train Phase 1 Mask2Map with 8 GPUs
```
./tools/dist_train.sh ./projects/configs/mask2map/M2M_nusc_r50_full_1Phase_12n12ep.py 8
```

Train Phase 2 Mask2Map with 8 GPUs (loads Phase 1 checkpoint automatically)
```
./tools/dist_train.sh ./projects/configs/mask2map/M2M_nusc_r50_full_2Phase_12n12ep.py 8
```

## Single-stage joint training (new)

Train Mask2Map end-to-end in a single run — no Phase 1 pre-training required.
Both the segmentation head (IMPNet) and the vectorization head (MMPNet) are
optimised jointly from epoch 1.

```
./tools/dist_train.sh ./projects/configs/mask2map/M2M_nusc_r50_full_joint_24ep.py 8
```

**Caveats:**
- Joint training from scratch typically converges slightly slower than the
  two-phase approach for the same total epoch budget.  The 24-epoch schedule
  (equivalent to 12 + 12 phases) is a reasonable starting point.
- If you observe training instability in early epochs, reduce `loss_pts` or
  `loss_cls` (vectorization weights) for the first few epochs, or increase
  `total_epochs` to 48.

## Eval Mask2Map with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/mask2map/M2M_nusc_r50_full_2Phase_12n12ep.py ./path/to/ckpts.pth 8
```

The same eval command works for the joint-training checkpoint:
```
./tools/dist_test_map.sh ./projects/configs/mask2map/M2M_nusc_r50_full_joint_24ep.py ./path/to/ckpts.pth 8
```