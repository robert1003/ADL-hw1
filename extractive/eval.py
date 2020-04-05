# setup environment
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# hyperparams
import sys
BATCH_SIZE = 256
TEST_FILE_PATH = sys.argv[1]
PREDICTION_FILE_PATH = sys.argv[2]
EMBEDDING_SAVE_PATH = 'word2vec_extractive.pickle'#'../embeddings/numberbatch-en-19.08.txt'
EMBEDDING_DIM = 300
MIN_DISCARD_LEN = 2

INPUT_LEN = 301

pretrained_ckpt = 'extractive/model_best_rouge1.ckpt'

device = 'cuda'

# read data
print('reading data...')
from _utils import read_jsonl
test_X, _, idx_X = read_jsonl(TEST_FILE_PATH, False, True)
print('done')

# load pretrained word embedding
print('loading word embedding...')
from _word2vec import Word2Vec
word2vec = Word2Vec(EMBEDDING_SAVE_PATH, 300, raw=False)
embedding = word2vec.embedding

SOS_token = word2vec.word2idx['<SOS>']
EOS_token = word2vec.word2idx['<EOS>']
PAD_token = word2vec.word2idx['<PAD>']
UNK_token = word2vec.word2idx['<UNK>']
print('done')

# transform sentences to embeddin4g
print('test_X')
test_X = word2vec.sent2idx(test_X, INPUT_LEN)

# convert them to dataset and dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader

test_dataset = TensorDataset(torch.from_numpy(test_X))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# build model
from _model import Model
from torch import optim
import torch.nn as nn

teacher_forcing_ratio = 0.5
GRAD_MAX = 1

model = Model(
    embedding=embedding,
    input_size=embedding.shape[0],
    hidden_size=embedding.shape[1],
    output_size=1, amp=1, n_layers=2, direction=2,
    dropout=0.0
).to(device)

# load model
print('loading pretrained model...')
checkpoint = torch.load(pretrained_ckpt)
model.load_state_dict(checkpoint['model_state_dict'])
print('done')

# define predict
import numpy as np
def predict(input_tensor):
    model.eval()
    
    batch_size = input_tensor.size(0)
    hidden = model.initHidden(batch_size)
    hidden = (hidden[0].to(device), hidden[1].to(device))
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    loss = 0
    prediction = []
    for i in input_tensor:
        output, hidden = model(i, hidden, batch_size)
        prediction.append(output.detach().cpu().numpy())
    
    return np.stack(prediction).swapaxes(0, 1)

# predict
print('predicting...')
predictions = []
for i, x in enumerate(test_loader):
    prediction = predict(x[0])
    predictions.append(prediction)
predictions = np.vstack(predictions)
print('done')

# print to file
import json
from _utils import postprocess, write_prediction

write_prediction(PREDICTION_FILE_PATH, postprocess(test_X, predictions, EOS_token), idx_X)
