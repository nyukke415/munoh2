# -*- coding: utf-8 -*-
# last updated:<2022/06/29/Wed 17:54:46 from:tuttle-desktop>

import torch
import torch.nn.functional as F

import os
import argparse

import munoh2
import utils
import ml_utils
import txtutils
import word2vec

def test(test, vocab, params, args):
    # load word2vec ################
    if params["word2vec"] == None:
        w2v = None
    else:
        w2v, vocab.w2i, vocab.i2w = word2vec.load_word2vec(params["word2vec"], vocab, params["vocab_size"])

    batch_size = 1
    model = munoh2.Munoh2(
        hidden_dim = params["hidden_dim"],
        word_embed = params["word_embed"],
        vocab_size = params["vocab_size"],
        batch_size = batch_size,
        vocab = vocab,
        word2vec = w2v,
        device = "cpu"
    )
    model.load_state_dict(
        torch.load(args.work_dir+"/"+args.model_name, map_location="cpu")
    )
    print("loaded:\t"+args.work_dir+"/"+args.model_name)

    model.eval()
    torch.no_grad()
    refs = list()
    hyps = list()

    user_prompt = "あなた: " if args.lang == "ja" else "You: "
    munoh_prompt = "むのう: " if args.lang == "ja" else "Munoh: "
    bye = "ばいばい！" if args.lang == "ja" else "Bye!"

    if args.interactive == True:
        print()
        while True:
            try:
                txt = input(user_prompt)
            except EOFError:
                print()
                print(munoh_prompt + bye)
                exit()
            txt = txtutils.tokenize(
                txtutils.normalize(txt, args.lang), args.lang
            )
            reply, attention_weight = model.reply(
                [txt],
                20
            )
            if args.lang == "ja":
                print(munoh_prompt, "".join(reply))
            elif args.lang == "en":
                print(munoh_prompt, " ".join(reply))

            if args.heatmap == True:
                ml_utils.plot_heatmap(attention_weight, txt.split(), reply)
    else:
        for i in range(0, len(test), batch_size):
            ref = test[i].get_reply().split(" ")
            hyp, attention_weight = model.reply(
                [test[i].get_txt()],
                20
            )
            test[i].hyp = hyp
            refs.append([ref])
            hyps.append(hyp)

            print("input:", test[i].get_txt())
            print("ref:", " ".join(ref))
            print("hyp:", " ".join(hyp))
            print("################")
            if args.heatmap == True:
                ml_utils.plot_heatmap(attention_weight, test[i].get_txt().split(), hyp)

        for i in range(1, 5):   # 1, 2, 3 ,4
            print("BLEU"+str(i), "{:.2f}".format(res[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=str, help="work directory")
    parser.add_argument("--model-name", required=True, type=str, help="model name")
    parser.add_argument("--lang", type=str, required=True, help="Language: ja or en")
    parser.add_argument("--interactive", action='store_true')
    parser.add_argument("--heatmap", action='store_true')
    args = parser.parse_args()

    params = ml_utils.load_params(args.work_dir+"/params.csv")


    # prepare training data ################
    print("preparing test data ...")
    corpus_path = args.work_dir+"/corpus"
    if not os.path.exists(corpus_path):
        corpus_path = "/".join(args.work_dir.split("/")[0:-1])+"/corpus"
    data = utils.load_pickle(corpus_path)
    print("corpus:", len(data.train), len(data.test), len(data.dev))

    # prepare vocabularies ################
    print("preparing vocabularies ...")
    vocab_path = args.work_dir+"/vocab"
    if not os.path.exists(vocab_path):
        vocab_path = "/".join(args.work_dir.split("/")[0:-1])+"/vocab"
    vocab = utils.load_pickle(vocab_path)
    vocab.w2i = {k: v for (k, v) in list(vocab.w2i.items())[:params["vocab_size"]]}
    vocab.i2w = {k: v for (k, v) in list(vocab.i2w.items())[:params["vocab_size"]]}
    print("vocab:", len(vocab.w2i.keys()))

    test(data.test, vocab, params, args)
