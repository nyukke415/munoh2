 # -*- coding: utf-8 -*-
# last updated:<2022/06/14/Tue 15:26:20 from:tuttle-desktop>

import os
import collections

import utils

class Vocabulary():
    def __init__(self, txts, pre_txts):
        self.w2i, self.c2i = self.make_vocabularies(
            txts, pre_txts
        )
        self.i2w = {i: w for (w, i) in self.w2i.items()}
        self.c2w = {i: c for (c, i) in self.c2i.items()}

    def make_vocabularies(self, txts, pre_txts, min_freq=1):
        print("making vocabulary...", flush=True)
        w2i = dict()
        w2i["<UNK>"] = 0
        w2i["<BOS>"] = 1
        w2i["<EOS>"] = 2
        # w2i["<EMP>"] = 1

        # add words using txt data ################
        words = list()
        for txt in list(set(txts)):
            words += txt.split()
        words = collections.Counter(utils.flatten(words))
        i = len(w2i.keys())
        for (w, freq) in words.most_common():
            if freq < min_freq: # 頻度 min_freq 以上の単語のみ登録
                break
            if w in w2i.keys(): # 追加済みの単語は飛ばす
                continue
            w2i[w] = i
            i += 1
        print("words in train txt", len(w2i.keys()))

        # add words using pretrain data ################
        words = list()
        for txt in list(set(pre_txts)):
            words += txt.split()
        words = collections.Counter(utils.flatten(words))
        i = len(w2i.keys())
        for (w, freq) in words.most_common():
            if freq < min_freq: # 頻度 min_freq 以上の単語のみ登録
                break
            if w in w2i.keys(): # txt で追加済みの単語は飛ばす
                continue
            w2i[w] = i
            i += 1
        print("words in train+pretrain txt", len(w2i.keys()))

        # character vocab ################
        chars = ""
        for txt in pre_txts + txts:
            chars += "".join(txt.split(" ")[1: -1])
        chars = collections.Counter(list(set(list(chars))))
        c2i = dict()
        c2i["<UNK>"] = 0
        c2i["<BOS>"] = 1
        c2i["<EOS>"] = 2
        i = len(c2i.keys())
        for (c, freq) in chars.most_common():
            c2i[c] = i
            i += 1

        print("chars in train+pretrain txt", len(c2i.keys()))
        print("DONE: make_vocabularies", flush=True)
        return w2i, c2i
