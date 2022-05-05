# -*- coding: utf-8 -*-
# last updated:<2022/05/05/Thu 20:21:06 from:tuttle-desktop>

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import argparse
import random
import math

import utils
import ml_utils

import DataManager
import Vocabulary
import munoh2

def train(train, dev, vocab, args):
    device = torch.device("cuda:"+ml_utils.select_gpu() if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print("device:\t", device, device_name)
    batch_size = args.batch_size
    epoch_size = args.epoch_size

    model = munoh2.munoh2(
        hidden_dim = args.hidden_dim,
        word_embd = args.word_embd,
        char_embd = args.char_embd,
        vocab_size = len(vocab.w2i.keys()),
        char_size = len(vocab.c2i.keys()),
        batch_size = batch_size,
        vocab = vocab,
        device = device
    )

    model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    print()
    print("corpus_size:", len(train))
    print("vocab_size:", len(vocab.w2i))
    print("early_stopping:", args.early_stopping)
    print(model)
    print()

    model.train()
    loss_list = list()
    dev_loss_list = list()
    dev_bleu_list = list()
    permutation = list(range(len(train)))
    dev_permutation = list(range(len(dev)))

    n_worse = 0
    for epoch in range(epoch_size):
        random.shuffle(permutation)
        loss_list.append(0)
        # train ################
        for i in range(0, len(train), batch_size):
            model.zero_grad()
            indices = permutation[i: i+batch_size]
            if batch_size != len(indices):
                continue
            loss, logits = model.forward(
                [" ".join(["<BOS>"]*10)]*batch_size,
                [train[idx].get_txt() for idx in indices],
            )
            loss.backward()
            optimizer.step()
            loss_list[-1] += loss.item()
        loss_list[-1] = loss_list[-1] / int(len(train)/batch_size)

        # dev ################
        model.eval()
        with torch.no_grad():
            random.shuffle(dev_permutation)
            dev_loss_list.append(0)
            dev_bleu_list.append(0)
            for i in range(0, len(dev), batch_size):
                dev_indices = dev_permutation[i: i+batch_size]
                if batch_size != len(dev_indices):
                    continue
                loss, logits = model.forward(
                    [" ".join(["<BOS>"]*10)]*batch_size,
                    [dev[idx].get_txt() for idx in dev_indices],
                )
                refs = model.logits2words(logits, vocab.i2w)
                print(" ".join(refs[0][0]))
                hyps = [dev[idx].get_txt().split(" ")[1: -1] for idx in dev_indices]
                dev_loss_list[-1] += loss.item()
                dev_bleu_list[-1] += ml_utils.BLEU(3, refs, hyps)
        dev_loss_list[-1] = dev_loss_list[-1] / int(len(dev) / batch_size)
        dev_bleu_list[-1] = dev_bleu_list[-1] / int(len(dev) / batch_size)
        model.train()

        print("epoch: "+str(epoch+1)+"/"+str(epoch_size)
              + "\tloss: "+f"{loss_list[-1]:.5f}"
              + "\tdev_loss: "+f"{dev_loss_list[-1]:.5f}"
              + "\tdev_bleu: "+f"{dev_bleu_list[-1]:.5f}",
              flush=True
        )
        ml_utils.plot_graph(
            range(epoch+1),
            [loss_list, dev_loss_list],
            [["train_loss", "r"], ["dev_loss", "g"]],
            args.work_dir+"/lc.png"
        )
        ml_utils.plot_graph(
            range(epoch+1),
            [dev_bleu_list],
            [["dev_bleu", "b"]],
            args.work_dir+"/lc_bleu.png"
        )

        # early stopping ################
        if args.early_stopping != None and 2 < len(dev_loss_list) and dev_loss_list[-2] < dev_loss_list[-1]:
            n_worse += 1
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] *= 0.5
            #     print("updated lr:", param_group["lr"])
            print("n_worse:\t"+str(n_worse))
            if args.early_stopping == n_worse:
                ml_utils.save_model(model, args.work_dir+"/model_early")

        # save model every __ epoch ################
        if (epoch+1) % 10 == 0:
            ml_utils.save_model(model, args.work_dir+"/model"+str(epoch+1))

        # save best model ################
        if max(dev_bleu_list) == dev_bleu_list[-1]:
            ml_utils.save_model(model, args.work_dir+"/model_bleu")
        if min(dev_loss_list) == dev_loss_list[-1]:
            ml_utils.save_model(model, args.work_dir+"/model_loss")

    print("End", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1, help="gpu id (use cpu if <= -1)")
    parser.add_argument("--work-dir", required=True, type=str, help="work directory")
    parser.add_argument("--hidden-dim", required=True, type=int, help="hidden dimension")
    parser.add_argument("--word_embd", type=int, help="word_embd dimension", default=300)
    parser.add_argument("--char_embd", type=int, help="char_embd dimension", default=300)
    parser.add_argument("--epoch-size", required=True, type=int, help="epoch size")
    parser.add_argument("--batch-size", required=True, type=int, help="batch size")
    parser.add_argument("--early-stopping", type=int, default=None, help="early stopping")
    args = parser.parse_args()

    # set seed ################
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # load training data ################
    print(args)
    data = DataManager.DataManager()
    data.load_txt("../data/wikitextja.txt.part", [0.8, 0.1, 0.1])

    # load vocabulary ################
    vocab_path = args.work_dir+"/vocab"
    vocab = Vocabulary.Vocabulary(
        [t.get_txt() for t in data.train], []
    )
    utils.save_pickle(vocab_path, vocab)
    # vocab = utils.load_pickle(vocab_path)

    train(data.train, data.dev, vocab, args)
