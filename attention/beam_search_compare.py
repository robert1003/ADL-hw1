# setup environment
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# hyperparams
import sys
BATCH_SIZE = 32
VALID_FILE_PATH = sys.argv[1]
SCORER_FILE_PATH = 'scorer/scorer_my.py'
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
valid_X, valid_Y = read_jsonl(VALID_FILE_PATH)
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
valid_Y = word2vec.sent2idx(valid_Y, INPUT_LEN)

# convert them to dataset and dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader

valid_dataset = TensorDataset(torch.from_numpy(valid_X), torch.from_numpy(valid_Y))

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

# define beam
from queue import PriorityQueue
class BeamNode(object):
    def __init__(self, hidden, prev, idx, score, length):
        self.hidden = hidden
        self.prev = prev
        self.idx = idx
        self.score = score
        self.length = length

    def eval(self):
        return self.score / self.length

    def __lt__(self, o):
        return self.eval() > o.eval()

def predict_beam(input_tensor, beam_size):
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
    for i in tqdm(range(batch_size)):
        start_node = BeamNode(decoder_hidden[:, i:i + 1, :].contiguous(), None, decoder_input[i, :].contiguous(), 0, 1)

        all_nodes = [start_node]
        now_nodes = [start_node]
        end_pq = PriorityQueue()

        for j in range(TARGET_LEN):
            if len(now_nodes) == 0:
                break

            pq = PriorityQueue()

            for node in now_nodes:
                input, hidden = node.idx, node.hidden
                output, hidden = model(input, hidden, 1, encoding=False, enc_outputs=enc_outputs[:, i:i + 1, :])
                topv, topi = output.data.topk(beam_size)
                for (score, idx) in zip(topv.detach().squeeze(0), topi.detach().squeeze(0)):
                    nxt_node = BeamNode(hidden, node, idx.unsqueeze(0), node.score + score, node.length + 1)
                    pq.put(nxt_node)

            now_nodes = []
            for _ in range(beam_size):
                assert pq.qsize() > 0
                node = pq.get()
                all_nodes.append(node)
                if node.idx == EOS_token or j == TARGET_LEN - 1:
                    end_pq.put(node)
                else:
                    now_nodes.append(node)

        assert end_pq.qsize() > 0
        best_node = end_pq.get()

        predict = [best_node.idx.cpu().numpy()[0]]
        while best_node.prev is not None:
            best_node = best_node.prev
            predict.append(best_node.idx.cpu().numpy()[0])
        predict = predict[-2::-1]

        t = 0
        while len(predict) < TARGET_LEN:
            t += 1
            assert t <= 1000
            predict.append(PAD_token)

        decoder_predict.append(np.array(predict))

    return np.stack(decoder_predict)

# get idx
np.random.seed(890108)
valid_idxs = sorted(np.random.permutation(len(valid_loader))[:100])

# predict
from tqdm import tqdm
print('predicting...')
pr1 = []
pr2 = []
idxs = []
j = 0
for i, (x, y) in tqdm(enumerate(valid_loader)):
    if j >= len(valid_idxs):
        break
    if i < valid_idxs[j]:
        continue
    pr1.append(predict(x))
    pr2.append(predict_beam(x, 3))
    idxs.append(range(2000000 + BATCH_SIZE * i, 2000000 + BATCH_SIZE * (i + 1)))
    j += 1
idxs = np.hstack(idxs)
pr1 = np.vstack(pr1)
pr2 = np.vstack(pr2)
print('done')

# get score
from _utils import write_prediction
import os
import json

write_prediction('pp1.jsonl', pr1, idxs, word2vec, False)
write_prediction('pp2.jsonl', pr2, idxs, word2vec, False)

ss = json.loads(os.popen(f'python3 {SCORER_FILE_PATH} pp1.jsonl {VALID_FILE_PATH}').read())
print('w/o beam search', ss)
ss = json.loads(os.popen(f'python3 {SCORER_FILE_PATH} pp2.jsonl {VALID_FILE_PATH}').read())
print('w/ beam search', ss)
