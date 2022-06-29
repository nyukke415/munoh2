# -*- coding: utf-8 -*-
# last updated:<2022/06/29/Wed 17:45:09 from:tuttle-desktop>

import re
import torch
import subprocess
import nltk
from nltk.translate.bleu_score import SmoothingFunction


# pad で埋めてサイズを maxlen に揃える
def fill(batchs, maxlen, pad=0):
    res = list()
    for batch in batchs:
        res.append(batch + [pad]*(maxlen - len(batch)))
    return res

# list to idx
def l2i(l, w2i):
    res = list()
    for elm in l:
        if type(elm) == str:
            res.append([w2i.get(w, 0) for w in elm.split(" ")])
        else:
            res.append(l2i(elm, w2i))
    return res

import matplotlib.pyplot as plt
def plot_graph(x, ys, opts, path):    # opts = [(label, color), ...]
    plt.switch_backend("agg")
    plt.figure()
    for (y, (l, c)) in zip(ys, opts):
        plt.plot(x, y, c+'-', label=l)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(path)

import numpy as np
import seaborn as sns
import pandas as pd
def plot_heatmap(heatmap, x, y):
    heatmap = heatmap# [:20]
    x = ["<BOS>"] + x + ["<EOS>"]
    # y = y + ["<EOS>"]# [1:21]
    df = pd.DataFrame(data=heatmap, columns=x, index=y)
    sns.heatmap(df, cmap="Blues", annot=True, square=True)
    plt.show()


def save_model(model, params, path):
    print("saved:", path)
    torch.save(
        model.state_dict(), path
    )

def save_params(params, path):
    output = ""
    for (k, v) in params.items():
        output += ",".join([str(k), str(v)]) + "\n"
    with open(path, "w") as f:
        f.write(output)
    print("saved params:", path)

def load_params(path):
    params = dict()
    with open(path, "r") as f:
        csv = f.read()
    for line in csv.strip().split("\n"):
        k, v = line.strip().split(",")
        try:
            params[k] = int(v)
        except:
            params[k] = v
    return params

# 0.1以下 ほとんど役に立たない
# 0.1~0.2 主旨を理解するのが困難である
# 0.2~0.3 主旨は明白であるが、文法上の重大なエラーがある
# 0.3~0.4 理解できる、適度な品質の翻訳
# 0.4~0.5 高品質な翻訳
# 0.5~0.6 非常に高品質で、適切かつ流暢な翻訳
# 0.6以上 人が翻訳した場合よりも高品質であることが多い
def BLEU(n, refs, hyps):
    if n == 1:
        weights = (1, 0, 0, 0)
    elif n == 2:
        weights = (1/2, 1/2, 0, 0)
    elif n == 3:
        weights = (1/3, 1/3, 1/3, 0)
    elif n == 4:
        weights = (1/4, 1/4, 1/4, 1/4)
    else:
        print("invalid input: n =", n)
    bleu = 100 * nltk.translate.bleu_score.corpus_bleu(
        refs, hyps, weights# , smoothing_function=SmoothingFunction().method3
    )
    return bleu

def get_gpuinfo():
    querys = ["index", "name", "memory.free"]
    outputs = list()
    for query in querys:
        outputs.append(
            subprocess.check_output(
                "/usr/bin/nvidia-smi --query-gpu="+query+" --format=csv, noheader",
                shell=True
            ).decode().strip().split("\n")[1: ]
        )
    outputs = list(zip(*outputs))
    gpuinfo = list()
    for output in outputs:
        gpuinfo.append(
            {key: val for key, val in zip(querys, output)}
        )
    return gpuinfo

def select_gpu(threshold=2000):  # threshold: int() MiB
    gpuinfo = get_gpuinfo()
    index_mem = [
        (gpu["index"], int(gpu["memory.free"].split(" ")[0])) for gpu in gpuinfo
    ]
    index_mem = sorted(index_mem, key=lambda x: x[1], reverse=True)
    if threshold < index_mem[0][1]:
        return str(index_mem[0][0])
    else:
        return None
