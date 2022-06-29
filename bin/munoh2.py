# -*- coding: utf-8 -*-
# last updated:<2022/06/29/Wed 16:49:21 from:tuttle-desktop>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim

import ml_utils

class EncoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, bidirectional, dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

    def forward(self, vec_txt):
        # h.size() => [num_layers*(1(LSTM) or 2(Bi-LSTM)), B, H]
        ehs, state = self.encoder(
            vec_txt,
        )
        if self.encoder.bidirectional == True:
            state = (state[0][0::2] + state[0][1::2], state[1][0::2] + state[1][1::2])
            ehs = (
                ehs[:,:,:self.hidden_dim] + ehs[:,:,self.hidden_dim:]
            ).contiguous()   # [B, L, H*2] ==> [B, L, H]
        return ehs, state

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, ehs, hs):
        s = torch.bmm(self.attn(ehs), hs.transpose(1, 2)) # [B, inpuL, outputL]
        return F.softmax(s, dim=1)

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, dropout):
        super(AttnDecoderRNN, self).__init__()
        self.attn = Attention(hidden_dim)
        self.reply_generator = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )
        self.concat = nn.Linear(hidden_dim*2, hidden_dim)
        self.hidden2vocab = nn.Linear(hidden_dim, vocab_size)

    def forward(self, vec_source, ehs, state):
        hs, state = self.reply_generator(vec_source, state)
        attention_weight = self.attn(ehs, hs)
        context_vec = torch.bmm(attention_weight.transpose(1, 2), ehs)

        hs = torch.cat([hs, context_vec], dim=2)
        hs = torch.tanh(self.concat(hs))
        logits = self.hidden2vocab(hs)
        return logits, state, attention_weight

class Munoh2(nn.Module):
    def __init__(self, hidden_dim, word_embed, vocab_size, batch_size, vocab, word2vec, device):
        super(Munoh2, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embed = word_embed
        self.embedding_dim = self.word_embed

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = 2
        self.vocab = vocab
        self.device = device

        self.word_embedding = nn.Embedding(self.vocab_size, self.word_embed)
        if word2vec != None:
            if self.vocab_size != word2vec.size()[0] or self.word_embed != word2vec.size()[1]:
                print("Error: got wrong dim from word2vec")
                exit()
            self.word_embedding.weight = word2vec
            self.word_embedding.weight.requires_grad = False

        self.encoder = EncoderRNN(self.embedding_dim, self.hidden_dim, self.num_layers, True, 0.1)
        self.decoder = AttnDecoderRNN(self.embedding_dim, self.hidden_dim, self.vocab_size, self.num_layers, 0.6)

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch_txts, batch_reply):
        ids_txt, _ = self.txt2ids(batch_txts)
        vec_txt = self.ids2vec(ids_txt)
        ehs, state = self.encoder(vec_txt)

        source, target, mask = self.calc_decoder_io(batch_reply)
        vec_source = self.ids2vec(source)
        logits, state, attention_weight = self.decoder(vec_source, ehs, state)

        return logits, target, mask, attention_weight

    def calc_decoder_io(self, batch_reply):
        ids_reply, mask = self.txt2ids(batch_reply)
        source = ids_reply[:, :-1]
        target = ids_reply[:, 1:]
        return source, target, mask

    def reply(self, inputs, maxlen):
        input_size = len(inputs)
        ids_txt, _ = self.txt2ids(inputs)
        vec_txt = self.ids2vec(ids_txt)
        ehs, state = self.encoder(vec_txt)

        word_ids = torch.tensor([[self.vocab.w2i.get("<BOS>", 0)]]*input_size).to(self.device)
        attention_weight = torch.empty(input_size, ids_txt.size()[1], 0).to(self.device)
        logits = torch.empty(input_size, 0, self.vocab_size).to(self.device)

        for _ in range(maxlen+2-1): # 2:<BOS> and <EOS>, -1: <BOS>以降の長さに合わせる
            decoder_input = self.ids2vec(word_ids[:,-1].view(-1, 1))
            logits_step, state, aw = self.decoder(decoder_input, ehs, state)
            logits = torch.cat([logits, logits_step], dim=1)
            attention_weight = torch.cat([attention_weight, aw], dim=2)

            word_id = torch.empty(0)
            for i, logit in enumerate(logits_step):
                word_id = torch.cat([word_id, torch.FloatTensor([self.logit2wids(logit)[0]])])
            word_ids = torch.cat([word_ids, word_id.view(-1, 1).long().to(self.device)], dim=1)

        words = self.wids2words(word_ids[0].tolist())
        attention_weight = (attention_weight*100)[0].transpose(0, 1).int().tolist()[:len(words)]

        return words, attention_weight

    def compute_loss(self, logits, target, mask):
        loss = self.criterion(
            logits.view(-1, self.vocab_size),
            target.contiguous().view(-1),
        )
        loss = loss.masked_fill(mask[:, 1:].contiguous().view(-1), 0)
        loss = torch.sum(loss) / torch.sum(1 - mask[:, 1:].float())

        return loss

    def txt2ids(self, batch_txts):
        maxlen = max([len(sent.split()) for sent in batch_txts]) + 2 # to add <BOS> and <EOS>
        ids = list()
        for sent in batch_txts:
            if len(sent) == 0:
                ids.append([self.vocab.w2i.get("<BOS>"), self.vocab.w2i.get("<EOS>")])
            else:
                ids.append(
                    [self.vocab.w2i.get("<BOS>")]
                    + [self.vocab.w2i.get(w, 0) for w in sent.split()]
                    + [self.vocab.w2i.get("<EOS>")]
                )
        # maskd ####
        mask_size = [id.index(self.vocab.w2i.get("<EOS>"))+1 for id in ids]
        mask = [
            [0]*ms + [1]*(maxlen - ms) for ms in mask_size
        ]
        mask = torch.BoolTensor(mask).to(self.device)
        # padding ####
        for id in ids:
            ids = [id + (maxlen-len(id))*[self.vocab.w2i.get("<EOS>")] for id in ids]
        ids = torch.LongTensor(ids).to(self.device)
        return ids, mask

    def ids2vec(self, batch_ids):
        # word_embedding ####
        vec = self.word_embedding(
            batch_ids
        )

        return vec

    def logits2words(self, logits):
        words = list()
        for logit in logits:
            inf = self.logit2wids(logit)
            try:
                inf = inf[0: inf.index(self.vocab.w2i["<EOS>"])]
            except:             # inf 内に <EOS> が無い場合
                pass
            words.append([
                [self.vocab.i2w[wid] for wid in inf]
            ])
        return words

    def logit2wids(self, logit):
        wids = list()
        _, word_ids = torch.sort(logit, descending=True)
        word_ids = word_ids.tolist() # sentenceLength * vocabSize
        for widl in word_ids:        # 単語ごとに候補を順に選択
            if widl[0] == self.vocab.w2i["<UNK>"]: # 確率が一番高い単語が<UNK>なら，次の単語を選択
                wids.append(widl[1])
            else:
                wids.append(widl[0])
        return wids             # [398, 398, 989, 1183, 1146, ...]

    def wids2words(self, wids):
        try:
            wids = wids[0: wids.index(self.vocab.w2i["<EOS>"])]
        except:             # inf 内に <EOS> が無い場合
            pass
        words = [self.vocab.i2w[wid] for wid in wids]
        return words[1:]        # <BOS>を省く
