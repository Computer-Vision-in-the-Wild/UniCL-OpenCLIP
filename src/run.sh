#!/bin/bash
export DIR="$(dirname "$(pwd)")"
source activate open_clip
export PYTHONPATH=${PYTHONPATH}:${DIR}

export base_path="./"
export train_data="${base_path}cifar-10-rand_2-shot_50.csv"  # change to your path to data
export val_data="${base_path}test_cifar-10.csv"  # change to your path to data
export val_dataset="cifar-10"

export report_to="wandb"
export csv_img_key="filepath"
export csv_caption_key="title"
export csv_label_key="label"
export metrics="accuracy"
export eval_type='elevater'  # ic or clip


export save_frequency=10
export eval_frequency=${save_frequency}
export batch_size=32  # 512
export eval_batch_size=32
export warmup=500
export lr=1e-5
export wd=0.05
export workers=1
export model="ViT-B-32"
export pretrained="openai"  # openai
export epochs=200
export csv_separator=","
export loss="CLIP" # or CLIP

export MASTER_PORT=6527
export CUDA_VISIBLE_DEVICES=0
export nproc_per_node=1

torchrun --nproc_per_node ${nproc_per_node} --master_port ${MASTER_PORT} -m training.main \
    --report-to ${report_to} \
    --train-data ${train_data} \
    --val-data ${val_data}  \
    --val_dataset ${val_dataset} \
    --csv-img-key ${csv_img_key} \
    --csv-caption-key ${csv_caption_key} \
    --loss ${loss} \
    --warmup ${warmup} \
    --batch-size ${batch_size} \
    --lr ${lr} \
    --wd ${wd} \
    --epochs ${epochs} \
    --workers ${workers} \
    --model ${model} \
    --csv-separator ${csv_separator} \
    --local-loss \
    --gather-with-grad \
    --pretrained ${pretrained} \
    --save-frequency ${save_frequency} \
    --eval_type ${eval_type} \
    --eval_frequency ${eval_frequency}

# export multi_label_pos_type="exact_same"  # contain_same, exact_same
# --multi_label_pos_type ${multi_label_pos_type} \