#!/bin/sh
dir=~/data/munoh2/model/train-movie
# model="model22"
model="model_bleu"
# model="model_loss"

python -u test.py \
       --work-dir $dir \
       --model-name $model \
       --lang en\
       --heatmap \
       2>&1 | tee ${dir}/test.log
