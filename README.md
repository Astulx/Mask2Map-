<div align="center">
<h1>[ECCV24] Mask2Map <img src="assets/map.png" width="30"></h1>
<h3>Vectorized HD Map Construction Using Bird’s Eye View Segmentation Masks</h3>

Sewhan Choi<sup>1</sup> \*, Jungho Kim<sup>1</sup> \*, Hongjae Shin<sup>1</sup>, Junwon Choi<sup>2</sup> \**
 
<sup>1</sup> Hanyang University, Korea <sup>2</sup> Seoul National University, Korea

(\*) equal contribution, (<sup>**</sup>) corresponding author.

ECCV papers ([ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08664.pdf))

ArXiv Preprint ([arXiv 2407.13517](https://arxiv.org/abs/2407.13517))
</div>

![overall](assets/overall.png "framework")

## Updates
* [2024-09] Release the model codes and trained weights of 24 and 110 epochs on the nuScenes dataset, respectively

## Upcomings
* [2024-10] Trained weights on the Argoverse2 dataset.
* [2024-10] Trained weights for camera and LiDAR fusion on the nuScenes dataset.

## Introduction
In this paper, we introduce Mask2Map, a novel end-to-end online HD map construction method designed for autonomous driving applications. Our approach focuses on predicting the class and ordered point set of map instances within a scene, represented in the bird's eye view (BEV).
Mask2Map consists of two primary components: the Instance-Level Mask Prediction Network (IMPNet) and the Mask-Driven Map Prediction Network (MMPNet). IMPNet generates Mask-Aware Queries and BEV Segmentation Masks to capture comprehensive semantic information globally. Subsequently, MMPNet enhances these query features using local contextual information through two submodules: the Positional Query Generator (PQG) and the Geometric Feature Extractor (GFE). PQG extracts instance-level positional queries by embedding BEV positional information into Mask-Aware Queries, while GFE utilizes BEV Segmentation Masks to generate point-level geometric features.
However, we observed limited performance in Mask2Map due to inter-network inconsistency stemming from different predictions to Ground Truth (GT) matching between IMPNet and MMPNet. To tackle this challenge, we propose the Inter-network Denoising Training method, which guides the model to denoise the output affected by both noisy GT queries and perturbed BEV Segmentation Masks.

## Models
> Results from the [Mask2Map paper](https://arxiv.org/abs/2308.05736)

## Qualitative results on nuScenes val split 

<div align="center"><h4> nuScenes dataset</h4></div>



| Method | Backbone | BEVEncoder |Lr Schd | mAP| config | Download_phase1 | Download_phase2 |
| :---: | :---: | :---: | :---: |  :---: |:----------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------:|
| Mask2Map| R50 |bevpool | 24ep | 71.6 | [config](https://github.com/SehwanChoi0307/Mask2Map/tree/main/projects/configs/mask2map/M2M_nusc_r50_full_2Phase_12n12ep.py) | [model_phase1](https://drive.google.com/file/d/1ntFC0yQmWmr1k9zKWfj4ii2Ty-bDH8js/view?usp=sharing) | [model_phase2](https://drive.google.com/file/d/1lEvVJvn5fLEd2qYNTnejvF_l4IdwXTt7/view?usp=sharing) | 
| Mask2Map| R50 |bevpool | 110ep | 75.4 | [config](https://github.com/SehwanChoi0307/Mask2Map/tree/main/projects/configs/mask2map/M2M_nusc_r50_full_2Phase_55n55ep.py) | [model_phase1](https://drive.google.com/file/d/1nuFMOmH8UqHW3FlreX19Uf8Il2ldKV-E/view?usp=sharing) | [model_phase2](https://drive.google.com/file/d/1eX17lKbqkLqOkk18u7vPIQhDvuT5j81O/view?usp=sharing) |

**Notes**: 
- All the experiments are performed on 4 NVIDIA GeForce RTX 3090 GPUs.

## Single-stage joint training (new)

A single-stage config is now available for users who want to skip the two-phase
workflow and train segmentation + vectorization jointly in one run:

```bash
./tools/dist_train.sh \
    projects/configs/mask2map/M2M_nusc_r50_full_joint_24ep.py 8
```

See [Train and Eval](docs/train_eval.md) for full details and caveats.

## Getting Started
- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Train and Eval](docs/train_eval.md)


## Query Initialization Modes

The 1-Phase transformer (`Mask2Map_Transformer_1Phase`) supports two modes for
initializing the decoder instance queries, controlled by the `query_init_type`
parameter in the transformer config dict.

| Value | Behaviour |
|---|---|
| `"learnable"` *(default)* | Queries are initialized from a fixed, learnable `nn.Embedding` table — the original Mask2Map behaviour. Existing checkpoints are fully compatible. |
| `"seg_guided"` | Queries are initialized from the highest-resolution BEV feature map produced by the pixel decoder.  The top-*K* spatially active positions (ranked by feature L2 norm) are selected and the corresponding feature vectors are gathered as initial query content.  A lightweight `Linear + LayerNorm` projection aligns the sampled features with the learnable-query embedding space.  Safe fallbacks prevent NaN / Inf and pad with learnable queries whenever fewer than *K* valid positions exist or batch items carry degenerate features. |

### How to enable seg-guided mode

Add `query_init_type="seg_guided"` to the `transformer` sub-dict in your
config file:

```python
transformer=dict(
    type="Mask2Map_Transformer_1Phase",
    ...
    query_init_type="seg_guided",   # <-- new option
    ...
)
```

A ready-to-use example config is provided at:
```
projects/configs/mask2map/M2M_nusc_r50_full_1Phase_12n12ep_seg_guided.py
```

Train with:
```bash
./tools/dist_train.sh \
    projects/configs/mask2map/M2M_nusc_r50_full_1Phase_12n12ep_seg_guided.py \
    <num_gpus>
```

### Caveats

* `query_init_type="seg_guided"` introduces a small additional `seg_query_proj`
  (`Linear + LayerNorm`) parameter group.  Existing `"learnable"` checkpoints
  cannot be loaded directly into a `"seg_guided"` model (and vice-versa).
* The seg-guided init is deterministic given the BEV features; no randomness is
  added at inference time.
* The `"learnable"` default is unchanged — all existing configs and
  checkpoints continue to work without modification.


## Demo

![demo](assets/demo.gif "demo")

## Acknowledgements

Mask2Map is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). It is also greatly inspired by the following outstanding contributions to the open-source community: [MapTR](https://github.com/hustvl/MapTR), [BEVFusion](https://github.com/mit-han-lab/bevfusion), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet), [GKT](https://github.com/hustvl/GKT), [VectorMapNet](https://github.com/Mrmoore98/VectorMapNet_code).

## Citation
If you find Mask2Map is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{Mask2Map,
  title={Mask2Map: Vectorized HD Map Construction Using Bird’s Eye View Segmentation Masks},
  author={Choi, Sewhan and Kim, Jungho and Shin, Hongjae and Choi, Jun Won},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
