# setup environment
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# hyperparams
BATCH_SIZE = 32
TRAIN_FILE_PATH = 'data/train.jsonl'
VALID_FILE_PATH = 'data/valid.jsonl'
TEST_FILE_PATH = 'data/test.jsonl'
EMBEDDING_FILE_PATH = 'embeddings/glove.840B.300d.txt'
EMBEDDING_SAVE_PATH = 'word2vec_attention.pickle'

GENERATIVE_SCORER_PATH = 'scorer/scorer_generative.py'
TEMP_FILE_PATH = 'attention/asdfsndfj.jsonl'
EMBEDDING_DIM = 300
MIN_DISCARD_LEN = 999999999

INPUT_LEN = 251
TARGET_LEN = 40

CKPT_VALID_NAME = 'attention/model_best_rouge1.ckpt'
CKPT_NAME = 'attention/model.ckpt'

device = 'cuda'

# set random seed
import torch
import random
import numpy as np
#torch.manual_seed(1)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True
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
embedding = word2vec.make_embedding([train_X, train_Y, valid_X, valid_Y, test_X], MIN_DISCARD_LEN)

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
print('train_Y')
train_Y = word2vec.sent2idx(train_Y, TARGET_LEN)
print('valid_X')
valid_X = word2vec.sent2idx(valid_X, INPUT_LEN)
print('valid_Y')
valid_Y = word2vec.sent2idx(valid_Y, TARGET_LEN)
print('test_X')
test_X = word2vec.sent2idx(test_X, INPUT_LEN)

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
optimizer = optim.Adam(model.parameters(), amsgrad=True)
criterion = nn.NLLLoss(ignore_index=PAD_token)

# define train, evaluate, predict
# train
import random

def train(input_tensor, target_tensor):
    model.train()
    
    batch_size = input_tensor.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size).to(device)
    
    optimizer.zero_grad()
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    enc_outputs, encoder_hidden = model(input_tensor, encoder_hidden, batch_size, encoding=True, enc_outputs=None)
    
    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    encoder_hidden = encoder_hidden.view(encoder_n_layers, encoder_direction, batch_size, encoder_hidden_size)
    decoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2)

    loss = 0
    for di in target_tensor:
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False, enc_outputs=enc_outputs)
        loss += criterion(decoder_output, di.view(-1))
        decoder_input = di
    loss.backward()
    
    #_ = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAX)

    optimizer.step()
    
    return loss.item() / target_length

# evaluate
def evaluate(input_tensor, target_tensor):
    model.eval()
    
    batch_size = input_tensor.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size).to(device)
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    enc_outputs, encoder_hidden = model(input_tensor, encoder_hidden, batch_size, encoding=True, enc_outputs=None)
    
    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    encoder_hidden = encoder_hidden.view(encoder_n_layers, encoder_direction, batch_size, encoder_hidden_size)
    decoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2)
    
    loss = 0
    for di in target_tensor:
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False, enc_outputs=enc_outputs)
        loss += criterion(decoder_output, di.view(-1))
        decoder_input = di
    
    return loss.item() / target_length

# predict
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
    for di in range(TARGET_LEN):
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False, enc_outputs=enc_outputs)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi.detach().to(device)
        
        decoder_predict.append(topi.cpu().numpy())
    
    return np.hstack(decoder_predict)

# train model
from _utils import timeSince, print_words, write_prediction
import random
import json
import time
import numpy as np

start = time.time()
epochs = 100
patience = 100
cnt = 0
iter_per_epoch = len(train_loader)
val_per_epoch = len(valid_loader)
#teacher_forcing_ratio = 1.0
#decay = 0.008
best_rogue1 = 0

for epoch in range(epochs):
    total_loss = 0
    tot = len(train_loader)
    for i, (x, y) in enumerate(train_loader):
        if i > iter_per_epoch:
            break
        loss = train(x, y)#, teacher_forcing_ratio)
        total_loss += loss
        print('epoch {}: {}/{}, loss={}'.format(epoch + 1, i, iter_per_epoch, loss), end='\r')
    print('epoch {}/{}: avg train loss={}'.format(epoch + 1, epochs, total_loss / iter_per_epoch))
    
    #teacher_forcing_ratio = 1.0 * np.exp(-decay * (epoch + 1))
    
    if (epoch + 1) % 1 == 0:
        total_loss = 0  # Reset every epoch
        valid_idxs = sorted(np.random.permutation(len(valid_loader))[:val_per_epoch])
        idxs = []
        j = 0
        tot = len(valid_loader)
        all_predict = []
        for i, (x, y) in enumerate(valid_loader):
            if j >= len(valid_idxs):
                break
            if i < valid_idxs[j]:
                continue
            loss = evaluate(x, y)
            all_predict.append(predict(x))
            total_loss += loss
            idxs.append(range(2000000 + BATCH_SIZE * i, 2000000 + BATCH_SIZE * (i + 1)))
            j += 1
            print('epoch {}: {}/{}, loss={}'.format(epoch + 1, j, val_per_epoch, loss), end='\r')
            
        idxs = np.hstack(idxs)
        #print(valid_idxs, idxs)
        all_predict = np.vstack(all_predict)
        write_prediction(TEMP_FILE_PATH, all_predict, idxs, word2vec)
        ss = json.loads(os.popen(f'python3 {GENERATIVE_SCORER_PATH} {TEMP_FILE_PATH} {VALID_FILE_PATH}').read())
        ss = json.loads(os.popen(f'python3 {GENERATIVE_SCORER_PATH} {TEMP_FILE_PATH} {VALID_FILE_PATH}').read())
        print('epoch {}/{}: avg valid loss={}'.format(epoch + 1, epochs, total_loss / val_per_epoch))
        
        print(ss)
    
        if ss['mean']['rouge-1'] > best_rogue1:
            print('update best_rogue1: {} -> {}, saving model...'.format(best_rogue1, ss['mean']['rouge-1']))
            best_rogue1 = ss['mean']['rouge-1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_rogue1': best_rogue1
            }, CKPT_VALID_NAME)
            cnt = 0
        else:
            cnt += 1
        
    print(timeSince(start, epoch + 1, epochs))
    '''
    tX, tY = random.choice(list(zip(train_X, train_Y)))
    pt = predict(torch.from_numpy(tX).view(1, -1))
    print_words('train', tX.reshape(1, -1), tY.reshape(1, -1), pt, f, word2vec)

    vX, vY = random.choice(list(zip(valid_X, valid_Y)))
    pv = predict(torch.from_numpy(vX).view(1, -1))
    print_words('valid', vX.reshape(1, -1), vY.reshape(1, -1), pv, f, word2vec)
    '''
    if (epoch + 1) % 1 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, CKPT_NAME)