# Munoh2
チャットボット

Seq2Seq
Encoder: Bi-LSTM
Decoder: LSTM + Attention

# Dataset
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

Supported language: Japanese, English

# Usage
- ディレクトリの作成、データセットのダウンロード

  cd munoh2
  mkdir data model

  https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
  からダウンロードした movie_lines.txt を munoh2/data 配下に置いて、

  python bin/moviediag2conv.py

- データセットの前処理，辞書の作成

  bin/preprocess.sh

- Word2Vec の学習 (not mandatory)

  bin/word2vec.sh

- チャットボットの事前学習 (not mandatory)

  bin/pretrain.sh

- チャットボットの学習

  bin/train.sh
  bin/train+premodel.sh (using pre-trained model)

- テスト

  bin/test.sh
  bin/talkwith.sh (interactive mode)
