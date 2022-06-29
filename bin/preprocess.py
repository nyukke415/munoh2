# -*- coding: utf-8 -*-
# last updated:<2022/06/29/Wed 17:49:56 from:tuttle-desktop>

import argparse
from collections import Counter

import DataManager
import Vocabulary
import utils

def main(args):
    data = DataManager.DataManager()
    if args.pretrain_data != "":
        print("loading pre-training data...", flush=True)
        data.load_pretrain(args.pretrain_data, [0.8, 0.1, 0.1])
    if args.train_data != "":
        print("loading training data...", flush=True)
        data.load_train(args.train_data, [0.8, 0.1, 0.1])
    print("pretrain:", len(data.pretrain), len(data.pretest), len(data.predev))
    print("train:", len(data.train), len(data.test), len(data.dev))
    utils.save_pickle(args.work_dir+"/corpus", data)

    vocab = Vocabulary.Vocabulary(
        [t.get_txt() for t in data.train] + [t.get_reply() for t in data.train],
        [t.get_reply() for t in data.pretrain]
    )
    utils.save_pickle(args.work_dir+"/vocab", vocab)


    calc_cover_rate(data, vocab)


def calc_cover_rate(data, vocab):
    txt_words = list()
    reply_words = list()
    for t in data.train:
        txt_words += t.get_txt().split()
        reply_words += t.get_reply().split()
    txt_words = Counter(txt_words)
    reply_words = Counter(reply_words)
    twlen = sum(txt_words.values())
    rwlen = sum(reply_words.values())

    for vr in [i/100 for i in range(100, 0, -5)]: # vocaburary reduction rate
        vl = int(len(vocab.w2i.items())*vr)
        wlist = list(vocab.w2i.keys())[:vl]
        wintxt_cnt = sum([txt_words[w] for w in wlist])
        winreply_cnt = sum([reply_words[w] for w in wlist])
        tcr = winreply_cnt/rwlen * 100
        rcr = wintxt_cnt/twlen * 100
        print(
            "vocab size: "+str(vl)+" ("+str(int(vr*100))+"%)     ",
            "(txt, reply) cover rate:", f'{tcr:.1f}%', f'{rcr:.1f}%',
            sep="\t", flush=True
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=str, help="work directory")
    parser.add_argument("--pretrain-data", type=str, default="", help="pre-training txt")
    parser.add_argument("--train-data", type=str, default="", help="training txt")
    args = parser.parse_args()
    main(args)
