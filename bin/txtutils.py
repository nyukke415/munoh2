# -*- coding: utf-8 -*-
# last updated:<2022/06/14/Tue 14:31:28 from:tuttle-desktop>

import Mykytea
import zenhan
import nltk

opt = ""                    # command line options
print("loading KyTea ...", opt, end="")
tok = Mykytea.Mykytea(opt)
print("DONE")


def tokenize(sent, lang="ja"):
    if lang == "ja":
        global opt
        global tok
        res = tok.getTagsToString(sent)
        res = [chunk.split("/")[0] for chunk in res.split()]
        res = list(filter(lambda w: len(w) != 0, res))
    elif lang == "en":
        res = nltk.word_tokenize(sent)
    return " ".join(res)

# 半角->全角, 句読点を統一
def normalize(s, lang="ja"):
    if lang == "ja":
        if len(s) == 0:
            return s
        if s[0] == "<" and s[-1] == ">": # タグの場合はスキップ
            return s
        # if s[0] == "＜" and s[-1] == "＞":
        #     return zenhan.z2h(s)
        # trans_table = str.maketrans({"、": "，", "。": "．"})
        s = " ".join([zenhan.h2z(w) for w in s.split()])
        # s = s.translate(trans_table)
    elif lang == "en":
        s = s.lower()
    return s
