# -*- coding: utf-8 -*-
# last updated:<2022/06/02/Thu 12:24:54 from:tuttle-desktop>

import re

import random
import Datum


class DataManager:
    pretrain = list()
    predev = list()
    pretest = list()
    train = list()
    dev = list()
    test = list()
    def __init__(self):
        pass

    def load_pretrain(self, reply_fname, ratios = [0.9, 0.05, 0.05]):
        with open(reply_fname, "r") as f:
            reply = f.readlines()
        txt = ["<EOS>"]*len(reply)
        data = list()
        for tsent, rsent in zip(txt, reply):
            data.append(Datum.Datum())
            data[-1].txt = tsent.strip()
            data[-1].reply = rsent.strip()
        self.pretrain, self.predev, self.pretest = self.divide_data(data, ratios)

    def load_train(self, conv_fname, ratios = [0.8, 0.1, 0.1]):
        with open(conv_fname, "r") as f:
            conv = f.read()
        data = list()
        for chunk in re.split("\n\n+", conv): # 2つ以上の連続した改行で区切る
            chunk = chunk.strip().split("\n")
            for i in range(len(chunk)-1):
                data.append(Datum.Datum())
                data[-1].txt = chunk[i]
                data[-1].reply = chunk[i+1]
        self.train, self.dev, self.test = self.divide_data(data, ratios)


    def divide_data(self, data, ratios):
        size = len(data)
        random.shuffle(data)
        train = list()
        dev = list()
        test = list()
        train += data[0: int(ratios[0]*size)]
        dev += data[int(ratios[0]*size): int(sum(ratios[:2])*size)]
        test += data[int(sum(ratios[:2])*size):]
        return train, dev, test
