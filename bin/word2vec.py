# -*- coding: utf-8 -*-
# last updated:<2022/06/29/Wed 17:51:37 from:tuttle-desktop>

import os
import torch
import torch.nn as nn
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np

import utils

def load_word2vec(path, vocab, max_vocab):
    print("loading word2vec...", flush=True)
    wv = KeyedVectors.load_word2vec_format(path, binary=True)
    w2i = {"<UNK>": 0}
    w2v = np.zeros((1, len(wv.vectors[0])))

    for w, wid in list(vocab.w2i.items())[1:]: # <UNK> を飛ばす
        if max_vocab == len(w2i.keys()):
            break
        if w not in wv.vocab.keys():
            continue
        w2i[w] = len(w2i.keys())
        w2v = np.append(w2v, [wv.vectors[wv.vocab[w].index]], axis=0)

    # max_vocab に満たない分を，学習済みword2vecの辞書から追加
    for w, m in list(wv.vocab.items()):
        if max_vocab <= len(w2i.keys()):
            break
        if w not in w2i.keys():
            w2i[w] = len(w2i.keys())
            w2v = np.append(w2v, [wv.vectors[wv.vocab[w].index]], axis=0)
    i2w = {i: w for w, i in w2i.items()}
    w2v = w2v.astype(np.float32)
    w2v = nn.Parameter(torch.from_numpy(w2v))

    return w2v, w2i, i2w

def train(sents, path):
    model = word2vec.Word2Vec(sents, size=300, min_count=2, window=5,iter=100)
    model.wv.save_word2vec_format(path, binary=True)
    print("saved:", path)

def test(path):
    wv = KeyedVectors.load_word2vec_format(path, binary=True)
    while True:
        try:
            word = input("word: ")
        except EOFError:
            exit()
        print(wv[word])
        print(wv.most_similar(positive=[word]))
        print()

if __name__ == "__main__":
    path = os.environ["HOME"]+"/data/munoh2/model/word2vec/word2vec_movie_100.pt"

    # test ################
    # test(path)
    # exit()

    # train ################
    data = utils.load_pickle(os.environ["HOME"]+"/data/munoh2/model/train-movie/corpus")

    sents = list()
    sents = [
        d.get_reply() for d in data.pretrain + data.train
    ] + [
        d.get_txt() for d in data.train
    ]

    sents = set(sents)
    sents = [["<BOS>"]+sent.split()+["<EOS>"] for sent in sents]

    print("train sentence:", len(sents))

    train(sents, path)
