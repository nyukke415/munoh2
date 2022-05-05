# -*- coding: utf-8 -*-
# last updated:<2022/02/01/Tue 10:49:35 from:tuttle-desktop>

import Mykytea
import zenhan

opt = ""                    # command line options
print("loading KyTea ...", opt, end="")
tok = Mykytea.Mykytea(opt)
print("DONE")
def tokenize(sent):
    global opt
    global tok
    res = tok.getTagsToString(sent)
    res = [chunk.split("/")[0] for chunk in res.split()]
    res = list(filter(lambda w: len(w) != 0, res))
    res = " ".join(res)
    return res

# 半角->全角, 句読点を統一
def normalize(s):
    if len(s) == 0:
        return s
    if s[0] == "<" and s[-1] == ">": # タグの場合はスキップ
        return s
    # if s[0] == "＜" and s[-1] == "＞":
    #     return zenhan.z2h(s)
    trans_table = str.maketrans({"、": "，", "。": "．"})
    s = " ".join([zenhan.h2z(w) for w in s.split()])
    s = s.translate(trans_table)
    return s
