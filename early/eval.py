# setup environment
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# hyperparams
import sys
BATCH_SIZE = 512
TEST_FILE_PATH = sys.argv[1]
PREDICTION_FILE_PATH = sys.argv[2]
EMBEDDING_FILE_PATH = '../word2vec.pickle'#'../embeddings/numberbatch-en-19.08.txt'
EMBEDDING_DIM = 300
MIN_DISCARD_LEN = 5

INPUT_LEN = 251
TARGET_LEN = 30

pretrained_ckpt = 'model.ckpt'

device = 'cuda'

# read data
print('reading data...')
from _utils import read_jsonl
test_X, _, idx_X = read_jsonl(TEST_FILE_PATH, False, True)
print('done')

# load pretrained word embedding
print('loading word embedding...')
from _word2vec import Word2Vec
import pickle

with open(EMBEDDING_FILE_PATH, 'rb') as f:
    word2vec = pickle.load(f)

embedding = word2vec.embedding

SOS_token = word2vec.word2idx['<SOS>']
EOS_token = word2vec.word2idx['<EOS>']
PAD_token = word2vec.word2idx['<PAD>']
UNK_token = word2vec.word2idx['<UNK>']
print('done')

# transform sentences to embedding
print('test_X')
test_X = word2vec.sent2idx(test_X, INPUT_LEN)

# convert them to dataset and dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader

test_dataset = TensorDataset(torch.from_numpy(test_X))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# build model
from _model import EncoderRNN, DecoderRNN, Seq2seq
from torch import optim
import torch.nn as nn

model = Seq2seq(
    embedding=embedding,
    encoder=EncoderRNN(embedding.shape[0], embedding.shape[1], amp=1, n_layers=2, direction=2, dropout=0.5),
    decoder=DecoderRNN(embedding.shape[1], embedding.shape[0], amp=1, n_layers=2, direction=1, dropout=0.5)
).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss() # ignore_index=PAD_token

# load model
print('loading pretrained model...')
checkpoint = torch.load(pretrained_ckpt)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = checkpoint['criterion']
print('done')

# define predict
import numpy as np
def predict(input_tensor):
    model.eval()

    batch_size = input_tensor.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size).to(device)

    input_tensor = input_tensor.transpose(0, 1).to(device)

    input_length = input_tensor.size(0)

    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    for ei in input_tensor:
        encoder_output, encoder_hidden = model(ei, encoder_hidden, batch_size, encoding=True)

    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    decoder_hidden = encoder_hidden.view(2, 2, batch_size, 300)[:, 0, :, :].contiguous()

    decoder_predict = []

    for _ in range(TARGET_LEN):
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi.detach().to(device)

        decoder_predict.append(topi.cpu().numpy())
        #loss += criterion(decoder_output, di.view(-1))

    return np.hstack(decoder_predict)

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
from _utils import trim
with open(PREDICTION_FILE_PATH, 'w') as f:
    for pred, idx in zip(predictions, idx_X):
        pred = trim(pred.reshape(1, -1), word2vec)
        pred = pred.replace('<PAD> ', '').replace(' <PAD>', '').replace('<UNK> ', '').replace(' <UNK>', '')
        output = {'id':idx, 'predict':pred}
        s = json.dumps(output)
        print(s, file=f)
