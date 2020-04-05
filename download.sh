#!/usr/bin/env bash

early=140.112.90.197:9763/hw1/early.zip
extractive=140.112.90.197:9763/hw1/extractive.zip
seq2seq=140.112.90.197:9763/hw1/seq2seq.zip
attention=140.112.90.197:9763/hw1/attention.zip

#wget "${early}" -O ./temp.zip
#unzip temp.zip
#mv model.ckpt early/model.ckpt
#rm temp.zip

wget "${extractive}" -O ./temp.zip
unzip temp.zip
mv model_best_rouge1.ckpt extractive/model_best_rouge1.ckpt
rm temp.zip

wget "${seq2seq}" -O ./temp.zip
unzip temp.zip
mv model.ckpt seq2seq/model.ckpt
rm temp.zip

wget "${attention}" -O ./temp.zip
unzip temp.zip
mv model_best_rouge1.ckpt attention/model_best_rouge1.ckpt
rm temp.zip
