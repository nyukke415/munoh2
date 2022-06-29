# -*- coding: utf-8 -*-
# last updated:<2022/06/29/Wed 17:52:59 from:tuttle-desktop>

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import datetime
import argparse
import random
import math

import utils
import ml_utils

import DataManager
import Vocabulary
import munoh2
import word2vec

def train(train, dev, vocab, args):
    device = torch.device("cuda:"+ml_utils.select_gpu() if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print("device:\t", device, device_name)
    batch_size = args.batch_size
    epoch_size = args.epoch_size

    # load word2vec ################
    if args.word2vec == None:
        w2v = None
    else:
        w2v, vocab.w2i, vocab.i2w = word2vec.load_word2vec(args.word2vec, vocab, args.vocab_size)
        utils.save_pickle(vocab_path+str(args.vocab_size), vocab)

    # def model ################
    params = {
        "hidden_dim": args.hidden_dim,
        "word_embed": args.word_embed,
        "vocab_size": len(vocab.w2i.keys()),
        "batch_size": batch_size,
        "word2vec": args.word2vec
    }
    ml_utils.save_params(params, args.work_dir+"/params.csv")

    model = munoh2.Munoh2(
        hidden_dim = params["hidden_dim"],
        word_embed = params["word_embed"],
        vocab_size = params["vocab_size"],
        batch_size = params["batch_size"],
        vocab = vocab,
        word2vec = w2v,
        device = device
    )

    if args.pretrained_model != None:
        model.load_state_dict(
            torch.load(args.pretrained_model, map_location="cpu")
        )
        print("loaded:", args.pretrained_model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print()
    print("corpus_size:", len(train))
    print("vocab_size:", len(vocab.w2i))
    print("early_stopping:", args.early_stopping)
    print(model)
    print(optimizer)
    print()

    loss_list = list()
    dev_loss_list = list()
    dev_bleu_list = list()
    permutation = list(range(len(train)))
    dev_permutation = list(range(len(dev)))

    n_worse = 0
    for epoch in range(epoch_size):
        print(datetime.datetime.now(), flush=True)
        model.train()
        random.shuffle(permutation)
        loss_list.append(0)
        # train ################
        ploss = list()           # loss for print
        for iter_cnt, i in enumerate(range(0, len(train), batch_size)):
            model.zero_grad()
            indices = permutation[i: i+batch_size]
            if batch_size != len(indices):
                continue
            batch_txts = [train[idx].get_txt() for idx in indices]
            batch_replys = [train[idx].get_reply() for idx in indices]
            logits, target, mask, attention_weight = model.forward(
                batch_txts,
                batch_replys
            )
            loss = model.compute_loss(logits, target, mask)
            loss.backward()
            optimizer.step()
            loss_list[-1] += loss.item()

            # print progress ####
            prog = 100*(i+batch_size)/len(train)
            print("epoch "+str(epoch+1)+"/"+str(epoch_size)+f"  {prog:.1f}%       \tloss:", loss.item(), flush=True)
        loss_list[-1] = loss_list[-1] / (iter_cnt+1)

        # dev ################
        print("validating...\n", flush=True)
        model.eval()
        with torch.no_grad():
            random.shuffle(dev_permutation)
            dev_loss_list.append(0)
            dev_bleu_list.append(0)
            for iter_cnt, i in enumerate(range(0, len(dev), batch_size)[:int(12800/batch_size)]):
                dev_indices = dev_permutation[i: i+batch_size]
                if batch_size != len(dev_indices):
                    continue
                batch_txts = [dev[idx].get_txt() for idx in dev_indices]
                batch_replys = [dev[idx].get_reply() for idx in dev_indices]
                logits, target, mask, attention_weight = model.forward(
                    batch_txts,
                    batch_replys
                )
                loss = model.compute_loss(logits, target, mask)

                # logits, attention_weight = model.reply(
                #     batch_txts,
                #     max([len(b.split()) for b in batch_replys])
                # )
                # source, target, mask = model.calc_decoder_io(batch_replys)

                # logits, attention_weight = model.reply(
                #     batch_txts,
                #     max([len(b.split()) for b in batch_replys])
                # )
                hyps = model.logits2words(logits)
                refs = [dev[idx].get_reply().split(" ") for idx in dev_indices]
                dev_loss_list[-1] += loss.item()
                dev_bleu_list[-1] += ml_utils.BLEU(3, hyps, refs)
        dev_loss_list[-1] = dev_loss_list[-1] / (iter_cnt+1)
        dev_bleu_list[-1] = dev_bleu_list[-1] / (iter_cnt+1)
        model.train()

        # 現時点でのモデルで生成した文を表示
        for i in range(min(batch_size, 50)):
            print("input:", batch_txts[i])
            print("ref:", " ".join(refs[i]))
            print("hyp:", " ".join(hyps[i][0]))

        model.train()
        print("\nepoch: "+str(epoch+1)+"/"+str(epoch_size)
              + "\tloss: "+f"{loss_list[-1]:.5f}"
              + "\tdev_loss: "+f"{dev_loss_list[-1]:.5f}"
              + "\tdev_bleu: "+f"{dev_bleu_list[-1]:.5f}",
              flush=True
        )

        prefix = ""
        if args.pretrain == True:
            prefix = "pre"
        ml_utils.plot_graph(
            range(epoch+1),
            [loss_list, dev_loss_list],
            [["train_loss", "r"], ["dev_loss", "g"]],
            args.work_dir+"/"+prefix+"lc.png"
        )
        ml_utils.plot_graph(
            range(epoch+1),
            [dev_bleu_list],
            [["dev_bleu", "b"]],
            args.work_dir+"/"+prefix+"lc_bleu.png"
        )


        # model.eval()
        # model.train(False)

        # early stopping ################
        if args.early_stopping != None and 2 < len(dev_loss_list) and dev_loss_list[-2] < dev_loss_list[-1]:
            n_worse += 1
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] *= 0.5
            #     print("updated lr:", param_group["lr"])
            print("n_worse:\t"+str(n_worse))
            if args.early_stopping == n_worse:
                ml_utils.save_model(model, params, args.work_dir+"/"+prefix+"model_early")

        # save model every __ epoch ################
        if (epoch+1) % 1 == 0:
            model_name = args.work_dir+"/"+prefix+"model"+str(epoch+1)
            ml_utils.save_model(model, params, model_name)

        # save best model ################
        if max(dev_bleu_list) == dev_bleu_list[-1]:
            model_name = args.work_dir+"/"+prefix+"model_bleu"
            ml_utils.save_model(model, params, model_name)
        if min(dev_loss_list) == dev_loss_list[-1]:
            model_name = args.work_dir+"/"+prefix+"model_loss"
            ml_utils.save_model(model, params, model_name)

    print("End", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1, help="gpu id (use cpu if <= -1)")
    parser.add_argument("--work-dir", required=True, type=str, help="work directory")
    parser.add_argument("--hidden-dim", required=True, type=int, help="hidden dimension")
    parser.add_argument("--word-embed", type=int, help="word_embed dimension", default=300)
    parser.add_argument("--vocab-size", type=int, help="vocab size", default=5000)
    parser.add_argument("--epoch-size", required=True, type=int, help="epoch size")
    parser.add_argument("--batch-size", required=True, type=int, help="batch size")
    parser.add_argument("--lr", required=True, type=float, help="learning late")
    parser.add_argument("--word2vec", type=str, default=None, help="pre-trained word2vec model path")
    parser.add_argument("--early-stopping", type=int, default=None, help="early stopping")
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--pretrained-model", type=str, help="pretrained model")
    args = parser.parse_args()

    # set seed ################
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # prepare corpus and vocab ################
    print(args)
    data = DataManager.DataManager()

    if args.pretrained_model != None: # train + pretrained-mode ####
        vocab_path = "/".join(args.work_dir.split("/")[0: -1]) + "/vocab"
        corpus_path = "/".join(args.work_dir.split("/")[0: -1]) + "/corpus"
    else:
        vocab_path = args.work_dir+"/vocab"
        corpus_path = args.work_dir+"/corpus"

    vocab = utils.load_pickle(vocab_path)
    vocab.w2i = {k: v for (k, v) in list(vocab.w2i.items())[:args.vocab_size]}
    vocab.i2w = {k: v for (k, v) in list(vocab.i2w.items())[:args.vocab_size]}
    data = utils.load_pickle(corpus_path)

    if args.pretrain == True:   # pretrain ####
        train(data.pretrain, data.predev, vocab, args)
    else:                       # train or train+pretrained-model ####
        train(data.train, data.dev, vocab, args)
