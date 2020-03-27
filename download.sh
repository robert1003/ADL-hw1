#!/usr/bin/env bash

early_ckpt=https://github.com/robert1003/ADL-Assignments/releases/download/hw1-early/model.ckpt
early_word2vec=https://github.com/robert1003/ADL-Assignments/releases/download/hw1-early/word2vec.pickle

wget "${early_ckpt}" -O ./early/model.ckpt
wget "${early_word2vec}" -O ./word2vec.pickle
