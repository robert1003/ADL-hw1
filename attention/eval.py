# setup environment
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# hyperparams
import sys
BATCH_SIZE = 128
TEST_FILE_PATH = sys.argv[1]
PREDICTION_FILE_PATH = sys.argv[2]
EMBEDDING_SAVE_PATH = 'word2vec_attention.pickle'#'../embeddings/numberbatch-en-19.08.txt'
EMBEDDING_DIM = 300
MIN_DISCARD_LEN = 'inf'

INPUT_LEN = 251
TARGET_LEN = 40

pretrained_ckpt = 'attention/model_best_rouge1.ckpt'

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

# transform sentences to embedding
print('test_X')
test_X = word2vec.sent2idx(test_X, INPUT_LEN)

# convert them to dataset and dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader

test_dataset = TensorDataset(torch.from_numpy(test_X))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    enc_outputs, encoder_hidden = model(input_tensor, encoder_hidden, batch_size, encoding=True, enc_outputs=None)

    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    encoder_hidden = encoder_hidden.view(encoder_n_layers, encoder_direction, batch_size, encoder_hidden_size)
    decoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2)

    decoder_predict = []
    for di in range(TARGET_LEN):
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False, enc_outputs=enc_outputs)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi.detach().to(device)

        decoder_predict.append(topi.cpu().numpy())

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
from _utils import trim, remove_dupes
with open(PREDICTION_FILE_PATH, 'w') as f:
    for pred, idx in zip(predictions, idx_X):
        pred = trim(pred.reshape(1, -1), word2vec)
        pred = pred.replace('<PAD> ', '').replace(' <PAD>', '').replace('<UNK> ', '').replace(' <UNK>', '')
        pred = remove_dupes(pred)
        output = {'id':str(idx), 'predict':pred}
        s = json.dumps(output)
        print(s, file=f)
