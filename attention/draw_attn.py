# setup environment
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# hyperparams
import sys
BATCH_SIZE = 32
VALID_FILE_PATH = sys.argv[1]
PIC_FILE_PATH = sys.argv[2]
EMBEDDING_SAVE_PATH = 'word2vec_attention.pickle'#'../embeddings/numberbatch-en-19.08.txt'
EMBEDDING_DIM = 300
MIN_DISCARD_LEN = 'inf'

INPUT_LEN = 251
TARGET_LEN = 40
bid, tid = 4, 8

pretrained_ckpt = 'attention/model_best_rouge1.ckpt'

device = 'cuda'

# read data
print('reading data...')
from _utils import read_jsonl
valid_X, _, idx_X = read_jsonl(VALID_FILE_PATH, False, True)
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

# transform sentences to embedding
print('valid_X')
valid_X = word2vec.sent2idx(valid_X, INPUT_LEN)

# convert them to dataset and dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader

valid_dataset = TensorDataset(torch.from_numpy(valid_X))
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# build model
from _model import EncoderRNN, DecoderRNN, Attention, Seq2seq
from torch import optim
import torch.nn as nn

encoder_input_size = embedding.shape[1]
encoder_hidden_size = 256
encoder_n_layers = 1
encoder_direction = 2
encoder_dropout = 0.0

attn_size = encoder_direction * encoder_hidden_size
decoder_input_size = embedding.shape[1]
decoder_hidden_size = 256 * 2
decoder_output_size = embedding.shape[0]
decoder_n_layers = 1
decoder_direction = 1
decoder_dropout = 0.0

model = Seq2seq(
    embedding=embedding,
    encoder=EncoderRNN(encoder_input_size, encoder_hidden_size, encoder_n_layers, encoder_direction, encoder_dropout),
    decoder=DecoderRNN(decoder_input_size, decoder_hidden_size, decoder_output_size, attn_size, decoder_n_layers, decoder_direction, decoder_dropout),
    attention = Attention(encoder_direction * encoder_hidden_size, decoder_n_layers * decoder_direction * decoder_hidden_size),
    dropout=0.5
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
    encoder_hidden = model.encoder.initHidden(batch_size).to(device)

    input_tensor = input_tensor.transpose(0, 1).to(device)
    input_length = input_tensor.size(0)

    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    enc_outputs, encoder_hidden = model(input_tensor, encoder_hidden, batch_size, encoding=True, enc_outputs=None)

    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    encoder_hidden = encoder_hidden.view(encoder_n_layers, encoder_direction, batch_size, encoder_hidden_size)
    decoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2)

    decoder_predict = []
    attn_weights = []
    for di in range(TARGET_LEN):
        decoder_output, decoder_hidden, weight = model(decoder_input, decoder_hidden, batch_size, encoding=False, enc_outputs=enc_outputs, return_attn=True)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi.detach().to(device)

        attn_weights.append(weight.detach().cpu().numpy())
        decoder_predict.append(topi.cpu().numpy())

    return np.hstack(decoder_predict), np.stack(attn_weights)

# draw pic
from _utils import idx2words
for i, x in enumerate(valid_loader):
    if i == bid:
        break
        
x = x[0]
pred, weig = predict(x)
weig = weig.squeeze(3)

xx = idx2words(x[tid:tid + 1].numpy(), word2vec)[0]
xidx = xx.index('<EOS>')
print('input', xx[:xidx])

yy = idx2words(pred[tid:tid + 1], word2vec)[0]
yidx = yy.index('<EOS>')
print('predict', yy[:yidx])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(111)
cax = ax.matshow(weig[:yidx, tid, :xidx])
fig.colorbar(cax)

ax.set_xticklabels(xx[:xidx], rotation=90, fontsize=15)
ax.set_yticklabels(yy[:yidx], fontsize='large')

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax.set_xlabel('text (processed)', fontsize=15)
ax.xaxis.set_label_position('top')
ax.set_ylabel('predicted summary', fontsize=15)

plt.savefig(PIC_FILE_PATH, bbox_inches='tight')
