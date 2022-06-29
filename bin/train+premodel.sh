#!/bin/sh
cd $(dirname $0)
dir=/home/tuttle/Desktop/model/pretrain/train
mkdir $dir

python3 train.py --work-dir $dir --hidden-dim 256 --epoch-size 100 --batch-size 64 \
        --pretrained-model $dir/../premodel_loss \
        2>&1 | tee ${dir}/train+premodel.log
