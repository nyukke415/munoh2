# -*- coding: utf-8 -*-
# last updated:<2022/05/05/Thu 20:28:51 from:tuttle-desktop>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class munoh2(nn.Module):
    def __init__(self, hidden_dim, word_embd, char_embd, vocab_size, char_size, batch_size, vocab, device):
        super(munoh2, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embd = word_embd
        self.char_embd = char_embd
        self.embedding_dim = word_embd

        self.vocab_size = vocab_size
        self.char_size = char_size
        self.batch_size = batch_size
        self.num_layers = 1
        self.vocab = vocab
        self.device = device

        self.vec_size = self.word_embd

        self.word_embedding = nn.Embedding(
            vocab_size, self.word_embd
        )
        self.char_embedding = nn.Embedding(
            char_size, self.char_embd
        )

        # self.hidden2valrep = nn.Linear(self.hidden_dim, self.char_embd)

        self.vec2hidden = nn.Linear(self.vec_size, self.hidden_dim)
        self.hidden2vec = nn.Linear(self.hidden_dim, self.vec_size)
        self.hidden2vocab = nn.Linear(self.hidden_dim*2, self.vocab_size)
        self.reply_generator = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            # dropout=0.2,
            # batch_first=True
            bidirectional=True
        )

    def forward(self, batch_txts, batch_reply):
        maxlen = max([len(sent.split()) for sent in batch_txts+batch_reply]) + 2
        vec_txt, _ = self.txt2vec(batch_txts, maxlen)
        vec_reply, ids_reply = self.txt2vec(batch_reply, maxlen)
        loss, logits = self.vec2logit(vec_txt, vec_reply, ids_reply)

        print("forward:", loss.item())
        return loss, logits


    def txt2vec(self, batch_txts, maxlen):
        ids = self.txt2ids(batch_txts, maxlen)
        # convert id to vec
        return self.word_embedding(
            torch.LongTensor(ids).to(self.device)
        ), torch.LongTensor(ids).to(self.device)
    def txt2ids(self, batch_txts, maxlen):
        ids = list()
        for sent in batch_txts:
            ids.append(
                [self.vocab.w2i.get("<BOS>")]
                + [self.vocab.w2i.get(w, 0) for w in sent.split()]
                + [self.vocab.w2i.get("<EOS>")]
            )
        # padding
        for id in ids:
            ids = [id + (maxlen-len(id))*[self.vocab.w2i.get("<EOS>")] for id in ids]
        return ids

    def vec2logit(self, vec_txt, vec_reply, ids_reply):
        h = self.vec2hidden(vec_txt)
        c = self.vec2hidden(vec_txt)
        hs, (h, c) = self.reply_generator(vec_reply)
        logits = self.hidden2vocab(hs)

        loss = F.cross_entropy(
            logits.view(self.batch_size, self.vocab_size, -1),
            ids_reply,
            reduction="mean",
        )
        return loss, logits

    def logits2words(self, logits, i2w):
        words = list()
        for logit in logits:
            word_ids = logit.argmax(dim=1).tolist()
            inf = [i2w.get(wid, 0) for wid in word_ids]
            try:
                inf = inf[0: inf.index("<EOD>")]
            except:             # inf 内に <EOD> が無い場合
                pass
            words.append([inf])
        return words
