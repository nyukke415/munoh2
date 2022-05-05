# -*- coding: utf-8 -*-
# last updated:<2022/02/01/Tue 10:50:51 from:tuttle-desktop>

import nltk
import re

import txtutils

def main():
    output = list()
    with open("../data/WikiText-JA/WikiText-JA.txt", "r") as f:
        txt = f.read()
    txt = re.sub("<block>", "", txt)
    for line in txt.split("\n"):
        line = line.strip()
        if len(line) < 2 or line[:2] == "==":
            continue
        sents = split_sents(line)
        for sent in sents:
            if "*" in sent:     # 特殊文字が含まれていた文はスキップ
                continue
            words = txtutils.tokenize(sent).split()
            output.append(
                txtutils.normalize(" ".join(words).strip())
            )

    spath = "../data/wikitextja.txt"
    with open(spath, "w") as f:
        f.write("\n".join(output))
    print("saved:\t"+spath)

def split_sents(txt):
    sent_detector = nltk.RegexpTokenizer(u'[^　！？。]*[！？。]')
    sents = sent_detector.tokenize(txt)
    return sents

if __name__ == "__main__":
    main()
