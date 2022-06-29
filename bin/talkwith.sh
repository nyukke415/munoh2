#!/bin/sh
dir=~/data/munoh2/model/train-movie
# model="model_loss"
# model="model232"
model="model_bleu"

python -u test.py \
       --work-dir $dir \
       --model-name $model \
       --lang en \
       --interactive
       # --heatmap \
