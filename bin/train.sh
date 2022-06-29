#!/bin/sh
cd $(dirname $0)
dir=~/data/munoh2/model/train-movie
mkdir $dir

python3 train.py --work-dir $dir --epoch-size 50000 --batch-size 128 \
        --lr 0.0001 \
        --early-stopping 5 \
        --word-embed 300 \
        --hidden-dim 1000 \
        --vocab-size 3000 \
        --word2vec ~/data/munoh2/model/word2vec/word2vec_movie_100.pt \
        2>&1 | tee ${dir}/train.log
