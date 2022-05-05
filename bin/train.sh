#!/bin/sh
mkdir /tmp/tmp
python3 train.py --work-dir /tmp/tmp --hidden-dim 256 --epoch-size 30 --batch-size 128
