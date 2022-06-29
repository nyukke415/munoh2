# -*- coding: utf-8 -*-
# last updated:<2022/06/29/Wed 17:49:39 from:tuttle-desktop>

# Data: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

import codecs
import re
from collections import Counter

import txtutils

def main():
    txtfile = "../data/movie_lines.txt"
    characters = set()
    lines = list()

    with codecs.open(txtfile, "r", "utf-8", "ignore") as f:
        txt = f.read()
    for line in txt.strip().split("\n"):
        lid, cid, mid, cname, utterance = line.split(" +++$+++ ")
        lid = int(lid[1:])
        characters.add(cname.lower().strip())
        utterance = re.sub("</?[a-zA-Z]>", "", utterance) # htmlタグを削除
        utterance = txtutils.tokenize(txtutils.normalize(utterance, "en"), "en")
        if 10 < len(utterance.split()): # __ 単語以上の発話は使わない
            continue
        if "mmm" in " ".join(utterance):
            continue
        lines.append(
            {"lid": lid, "cid": cid, "mid": mid, "cname": cname, "utterance": utterance}
        )

    print("loaded:", txtfile)


    print("sent cnt:", len(lines))
    minfreq = 5
    vocab = Counter(sum([line["utterance"].split() for line in lines], []))
    wlist = list()
    for (w, f) in vocab.most_common():
        if f < minfreq:
            break
        wlist.append(w)
    wlist = set(wlist)

    # wlist に含まれる単語だけで構成された文のみ残す
    lines = [
        line for line in lines if len(set(line["utterance"].split())-wlist) == 0
    ]
    print("sent cnt:", len(lines))

    res = list()
    lines.sort(key=lambda x: x["lid"])
    for i in range(len(lines)-1):
        if lines[i]["lid"]+1 == lines[i+1]["lid"]: # 行番号が1違いなら
            txt = lines[i]["utterance"]
            reply = lines[i+1]["utterance"]
            if 1 < len(set(txt.split()+reply.split()) & characters):
                continue
            res.append((txt, reply))

    res = "\n".join(
        ["\n".join(r)+"\n" for r in res]
    ).strip()

    with open("../data/movie.txt", "w") as f:
        f.write(res)


if __name__ == "__main__":
    main()
