#!/usr/bin/env bash

word2vec=140.112.90.197:9763/word2vec.zip
early=140.112.90.197:9763/early.zip
seq2seq=140.112.90.197:9763/seq2seq.zip

wget "${word2vec}" -O ./ temp.zip
unzip temp.zip
rm temp.zip

#wget "${early}" -O ./ temp.zip
#unzip temp.zip
#mv model.ckpt early/model.ckpt
#rm temp.zip

wget "${seq2seq}" -O ./ temp.zip
unzip temp.zip
mv model.ckpt seq2seq/model.ckpt
rm temp.zip
