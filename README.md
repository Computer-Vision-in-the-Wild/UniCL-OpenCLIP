# UniCL-OpenCLIP

This is an open source implementation of Microsoft's UniCL:

["**Unifiled Contrastive Learning in Image-Text-Label Space. CVPR 2022**"](https://arxiv.org/abs/2204.03610) by [Jianwei Yang*](https://jwyang.github.io/), [Chunyuan Li*](https://chunyuan.li/), [Pengchuan Zhang*](https://pzzhang.github.io/pzzhang/), [Bin Xiao*](https://www.microsoft.com/en-us/research/people/bixi/), [Ce Liu](http://people.csail.mit.edu/celiu/), [Lu Yuan](https://scholar.google.com/citations?user=k9TsUVsAAAAJ&hl=en) and [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fjfgao%2F).

The codebased is forked from [open_clip](https://github.com/mlfoundations/open_clip).

## Introduction
<p align="center">
  <img src="figures/unified_cv.png" width=98%/>
</p>

In this paper, we introduce a new perspective on commonly used image-label and image-text data by residing them in an image-text-label space. In this space, a new learning paradigm, called **Unified Contrastive Learning (UniCL)** with a single learning objective is proposed to seamlessly prompt the synergy of two data types. We demonstrate that UniCL is an effective way of learning **semantically rich yet discriminative representations**, universally for image recognition in zero-shot, linear-probe, fully finetuning and transfer learning scenarios. When scaled up to billions of data, UniCL can exclusively learn a powerful visual-semantic representation supporting dozens of downstream tasks shown in [Florence](https://arxiv.org/pdf/2111.11432v1.pdf).

We make the comparisons between UniCL with coventional learning methods below:

<p align="center">
  <img src="figures/unicl_comparison.png" width=98%/>
</p>

## Getting Started

### Installation
Please follow [open_clip](https://github.com/mlfoundations/open_clip) for setting up environment, downloading datasets.

### Data preparation

To prepare datasets in [ELEVATER benchmark](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC), please follow [this instruction](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC).

To prepare ImageNet-1K dataset, please follow [DATA.md](https://github.com/microsoft/UniCL/blob/main/DATA.md).

To prepare Conceptual Captions dataset, YYCC or other datasets, please follow [this instruction](https://github.com/microsoft/UniCL/blob/main/DATA.md).

To see some examples of csv format dataset for UniCL, please check [this page]().


### Train UniCL

Please follow [this instruction](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC).

You can specify the loss function with `--loss UniCL` or `--loss CLIP`.

**Single-process running code**

```
python -m training.main \
  --train-data "/path/to/train_data.csv" \
    --val-data "/path/to/test_data.csv"  \
    --val_dataset "cifar-10" \
    --loss UniCL \
    --report-to wandb \
    --csv-img-key filepath \
    --csv-caption-key  title \
    --csv-label-key labels \
    --csv-separator "," \
    --warmup 500 \
    --batch-size 32 \
    --lr 1e-5 \
    --wd 0.05 \
    --epochs 200 \
    --workers 1 \
    --model "ViT-B-32" \
    --local-loss \
    --gather-with-grad \
    --pretrained "openai" \
    --save-frequency 10 \
    --metrics "accuracy" \
    --eval_type "elevater" \
    --eval_frequency 10
```

**SLURM**

To train the UniCL using **SLURM**, you may run `bash run.sh`. This also supports multi-gpu running.

### Evaluate

This repo also supports evaluation on downstream Image Classification dataset in [ELEVATER benchmark](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC). 

To use the evaluation method in ELEVATER, you first need to specify `--eval_type elevater`. Then, specify `--metrics` for one of the four types of metrics: `accuracy`, `11point_mAP`, `roc_auc` and `mean-per-class`; and specify the evaluation dataset, e.g. `--val_dataset cifar-10`. Make sure that your specified dataset name is one of the followings: `food-101`, `oxford-iiit-pets`, `resisc45_clip`, `mnist`, `kitti-distance`, `oxford-flower-102`, `gtsrb`, `cifar-100`, `patch-camelyon`, `stanford-cars`, `fgvc-aircraft-2013b-variants102`, `fer-2013`, `rendered-sst2`, `dtd`, `country211`, `cifar-10`, `caltech-101`, `eurosat_clip`, `hateful-memes`, `voc-2007-classification`.

You can customize your dataset by adding new templates and label names to `datasets/prompts.py` and specify your customized dataset name by `--val_dataset <your dataset name>`.

To only evaluate but not to train, do can just ignore adding `--train-data`.

Right now, this repo does not support adding external knowledge ([K-lite](https://github.com/Computer-Vision-in-the-Wild/klite)), and not support multi-label dataset `voc-2007-classification`. We are working on these!


## Citation

If you find this repo useful to your project, please consider to cite UniCL or/ and ELEVATER with the folling bibs:

```bibtex
@misc{yang2022unified,
    title={Unified Contrastive Learning in Image-Text-Label Space}, 
    author={Jianwei Yang and Chunyuan Li and Pengchuan Zhang and Bin Xiao and Ce Liu and Lu Yuan and Jianfeng Gao},
    year={2022},
    eprint={2204.03610},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

and

```bibtex
@article{li2022elevater,
    title={ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models},
    author={Li, Chunyuan and Liu, Haotian and Li, Liunian Harold and Zhang, Pengchuan and Aneja, Jyoti and Yang, Jianwei and Jin, Ping and Lee, Yong Jae and Hu, Houdong and Liu, Zicheng and Gao, Jianfeng},
    journal={Neural Information Processing Systems},
    year={2022}
}
```
