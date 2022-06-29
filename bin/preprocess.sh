#!/bin/sh
cd $(dirname $0)

# for pretrain or pretrain+train ####
# dir=~/data/munoh2/model/pretrain
# python3 preprocess.py --work-dir $dir \
#         --train-data ~/data/munoh2/data/girlschannel.txt \
#         --pretrain-data ~/data/munoh2/data/wikitextja.txt \
#         2>&1 | tee ${dir}/preprocess.log


# for train (without pretrain) ####
# dir=~/data/munoh2/model/train
# python3 preprocess.py --work-dir $dir \
#         --train-data ~/data/munoh2/data/girlschannel.txt \
#         2>&1 | tee ${dir}/preprocess.log



# for train (without pretrain) ####
dir=~/data/munoh2/model/train-movie
python3 preprocess.py --work-dir $dir \
        --train-data ~/data/munoh2/data/movie.txt \
        2>&1 | tee ${dir}/preprocess.log
