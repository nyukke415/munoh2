#!/bin/sh
cd $(dirname $0)
dir=~/data/munoh2/model/pretrain
mkdir $dir

python3 train.py \
        --work-dir $dir \
        --hidden-dim 256 \
        --epoch-size 10 \
        --batch-size 32 \
        --pretrain \
        --word2vec ~/data/munoh2/model/word2vec/word2vec_iter100.pt \
        2>&1 | tee -a ${dir}/pretrain.log
