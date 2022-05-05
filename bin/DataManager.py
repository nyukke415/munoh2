# -*- coding: utf-8 -*-
# last updated:<2022/05/05/Thu 10:58:53 from:tuttle-desktop>

import random
import Datum

class DataManager:
    train = list()
    dev = list()
    test = list()
    def __init__(self):
        pass

    def load_txt(self, fname, ratios = [0.8, 0.1, 0.1]):
        with open(fname, "r") as f:
            txt = f.readlines()
        txt = [Datum.Datum(sent.strip()) for sent in txt]
        random.shuffle(txt)
        size = len(txt)
        self.train = txt[0: int(ratios[0]*size)]
        self.dev = txt[int(ratios[0]*size): int(sum(ratios[:2])*size)]
        self.test =txt[int(sum(ratios[:2])*size):]
