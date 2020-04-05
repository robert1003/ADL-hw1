# setup environment
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# set hyperparams
BATCH_SIZE = 256
TRAIN_FILE_PATH = 'data/train.jsonl'
VALID_FILE_PATH = 'data/valid.jsonl'
TEST_FILE_PATH = 'data/test.jsonl'
TEMP_FILE_PATH = 'extractive/ncj2adA.jsonl'
EXTRACTIVE_SCORER_PATH = 'scorer/scorer_extractive.py'
EMBEDDING_FILE_PATH = 'embeddings/numberbatch-en-19.08.txt'
EMBEDDING_SAVE_PATH = 'word2vec_extractive.pickle'
EMBEDDING_DIM = 300
MIN_DISCARD_LEN = 2

INPUT_LEN = 301

GRAD_MAX = 1
CKPT_VALID_NAME = 'extractive/model_best_rouge1.ckpt'
CKPT_NAME = 'extractive/model.ckpt'
device = 'cuda'

import random, torch
import numpy as np

# fix random seed
# didn't fix torch seeds because it makes training very slow...
#torch.manual_seed(1)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
random.seed(881003)
np.random.seed(890108)

# read data
print('reading data...')
from _utils import read_jsonl
train_X, train_Y = read_jsonl(TRAIN_FILE_PATH)
valid_X, valid_Y = read_jsonl(VALID_FILE_PATH)
test_X, _ = read_jsonl(TEST_FILE_PATH, False)
print('done')

# load pretrained word embedding
print('loading word embedding...')
from _word2vec import Word2Vec
word2vec = Word2Vec(EMBEDDING_FILE_PATH, EMBEDDING_DIM)
embedding = word2vec.make_embedding([train_X, valid_X, test_X], MIN_DISCARD_LEN)

SOS_token = word2vec.word2idx['<SOS>']
EOS_token = word2vec.word2idx['<EOS>']
PAD_token = word2vec.word2idx['<PAD>']
UNK_token = word2vec.word2idx['<UNK>']
print('done')

# dump word2vec object
import pickle
with open(EMBEDDING_SAVE_PATH, 'wb') as f:
    tmp = {}
    tmp['embedding'] = word2vec.embedding
    tmp['word2idx'] = word2vec.word2idx
    tmp['idx2word'] = word2vec.idx2word
    pickle.dump(tmp, f)

# transform sentences to embedding
print('train_X')
train_X = word2vec.sent2idx(train_X, INPUT_LEN)
print('valid_X')
valid_X = word2vec.sent2idx(valid_X, INPUT_LEN)
print('test_X')
test_X = word2vec.sent2idx(test_X, INPUT_LEN)

# pad target to the same length of input
for i, _ in enumerate(train_Y):
    while len(train_Y[i]) < INPUT_LEN:
        train_Y[i].append(0)
    if len(train_Y[i]) > INPUT_LEN:
        train_Y[i] = train_Y[i][:INPUT_LEN]

    train_Y[i] = np.array(train_Y[i])
for i, _ in enumerate(valid_Y):
    while len(valid_Y[i]) < INPUT_LEN:
        valid_Y[i].append(0)
    if len(valid_Y[i]) > INPUT_LEN:
        valid_Y[i] = valid_Y[i][:INPUT_LEN]

    valid_Y[i] = np.array(valid_Y[i])

train_Y = np.array(train_Y)
valid_Y = np.array(valid_Y)

# check shape
print('shapes', train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape, test_X.shape)

# convert them to dataset and dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
valid_dataset = TensorDataset(torch.from_numpy(valid_X), torch.from_numpy(valid_Y))
test_dataset = TensorDataset(torch.from_numpy(test_X))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
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

optimizer = optim.Adadelta(model.parameters())
weight = torch.ones(1).to(device)
weight[0] = train_Y.shape[1] 
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

# define train, evaluate, predict
# train
def train(input_tensor, target_tensor):
    model.train()
    
    batch_size = input_tensor.size(0)
    hidden = model.initHidden(batch_size)
    hidden = (hidden[0].to(device), hidden[1].to(device))
    
    optimizer.zero_grad()
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    loss = 0
    for i, j in zip(input_tensor, target_tensor):
        output, hidden = model(i, hidden, batch_size)
        loss += criterion(output.squeeze(1), j.float())
        
    loss.backward()
    
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAX)
    
    optimizer.step()
    
    return loss.item() / target_length

# evaluate
def evaluate(input_tensor, target_tensor):
    model.eval()

    batch_size = input_tensor.size(0)
    hidden = model.initHidden(batch_size)
    hidden = (hidden[0].to(device), hidden[1].to(device))

    optimizer.zero_grad()

    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    loss = 0
    for i, j in zip(input_tensor, target_tensor):
        output, hidden = model(i, hidden, batch_size)
        loss += criterion(output.squeeze(1), j.float())

    return loss.item() / target_length

# predict
def predict(input_tensor):
    model.eval()

    batch_size = input_tensor.size(0)
    hidden = model.initHidden(batch_size)
    hidden = (hidden[0].to(device), hidden[1].to(device))

    optimizer.zero_grad()

    input_tensor = input_tensor.transpose(0, 1).to(device)

    input_length = input_tensor.size(0)

    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    loss = 0
    prediction = []
    for i in input_tensor:
        output, hidden = model(i, hidden, batch_size)
        prediction.append(output.detach().cpu().numpy())

    return np.stack(prediction).swapaxes(0, 1)

# train model
from _utils import timeSince, postprocess, write_prediction
import time, random, os, json
import numpy as np
start = time.time()
epochs = 100
patience = 100
cnt = 0
best_rogue1 = 0

for epoch in range(epochs):
    total_loss = 0
    tot = len(train_loader)
    for i, (x, y) in enumerate(train_loader):
        loss = train(x, y)
        total_loss += loss
        print('epoch {}: {}/{}, loss={}'.format(epoch + 1, i, tot, loss), end='\r')
    print('epoch {}/{}: avg train loss={}'.format(epoch + 1, epochs, total_loss / tot))
    
    total_loss = 0  # Reset every epoch
    all_predict = []
    tot = len(valid_loader)
    for i, (x, y) in enumerate(valid_loader):
        loss = evaluate(x, y)
        all_predict.append(predict(x))
        total_loss += loss
        print('epoch {}: {}/{}, loss={}'.format(epoch + 1, i, tot, loss), end='\r')
    print('epoch {}/{}: avg valid loss={}'.format(epoch + 1, epochs, total_loss / tot))
    print(timeSince(start, epoch + 1, epochs))
    
    all_predict = np.vstack(all_predict)
    pp = postprocess(valid_X, all_predict, EOS_token)
    write_prediction(f'{TEMP_FILE_PATH}', pp, list(range(2000000, 2020000)))
    ss = json.loads(os.popen(f'python3 {EXTRACTIVE_SCORER_PATH} {TEMP_FILE_PATH} {VALID_FILE_PATH}').read())
    
    if ss['mean']['rouge-1'] > best_rogue1:
        print('update best_rogue1: {} -> {}, saving model...'.format(best_rogue1, ss['mean']['rouge-1']))
        best_rogue1 = ss['mean']['rouge-1']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion': criterion,
            'best_rogue1': best_rogue1
        }, CKPT_VALID_NAME)
        cnt = 0
    else:
        cnt += 1
    
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion': criterion
        }, CKPT_NAME)

    if cnt > patience:
        print('done training')
        break
