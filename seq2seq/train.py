# setup environment
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# hyperparams
BATCH_SIZE = 128
TRAIN_FILE_PATH = 'data/train.jsonl'
VALID_FILE_PATH = 'data/valid.jsonl'
TEST_FILE_PATH = 'data/test.jsonl'
EMBEDDING_FILE_PATH = 'embeddings/numberbatch-en-19.08.txt'
EMBEDDING_SAVE_PATH = 'word2vec_seq2seq.pickle'
EMBEDDING_DIM = 300
MIN_DISCARD_LEN = 5

INPUT_LEN = 251
TARGET_LEN = 30

teacher_forcing_ratio = 0.5
GRAD_MAX = 5
CKPT_VALID_NAME = 'seq2seq/model_best_loss.ckpt'
CKPT_NAME = 'seq2seq/model.ckpt'

device = 'cuda'

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
from _model import EncoderRNN, DecoderRNN, Seq2seq
from torch import optim
import torch.nn as nn

model = Seq2seq(
    embedding=embedding,
    encoder=EncoderRNN(embedding.shape[0], embedding.shape[1], amp=1, n_layers=2, direction=2, dropout=0.5),
    decoder=DecoderRNN(embedding.shape[1], embedding.shape[0], amp=1, n_layers=2, direction=1, dropout=0.5)
).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss(ignore_index=PAD_token)

# define train, evaluate, predict
# train
import random

def train(input_tensor, target_tensor):
    model.train()
    
    batch_size = input_tensor.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size, device)
    
    optimizer.zero_grad()
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    loss = 0
    for ei in input_tensor:
        encoder_output, encoder_hidden = model(ei, encoder_hidden, batch_size, encoding=True)
        
    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    decoder_hidden = tuple(map(lambda f: f.view(2, 2, batch_size, 300)[:, 0, :, :].contiguous(), encoder_hidden))
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    for di in target_tensor:
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False)
        if random.random() < teacher_forcing_ratio:
            loss += criterion(decoder_output, di.view(-1))
            decoder_input = di
        else:
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.detach().to(device)
            loss += criterion(decoder_output, di.view(-1))

    loss.backward()
    
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_MAX)
    
    optimizer.step()
    
    return loss.item() / target_length

# evaluate
def evaluate(input_tensor, target_tensor):
    model.eval()
    
    batch_size = input_tensor.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size, device)
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    loss = 0
    for ei in input_tensor:
        encoder_output, encoder_hidden = model(ei, encoder_hidden, batch_size, encoding=True)
        
    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    decoder_hidden = tuple(map(lambda f: f.view(2, 2, batch_size, 300)[:, 0, :, :].contiguous(), encoder_hidden))
    
    for di in target_tensor:
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi.detach().to(device)

        loss += criterion(decoder_output, di.view(-1))

    return loss.item() / target_length

# predict
def predict(input_tensor):
    model.eval()
    
    batch_size = input_tensor.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size, device)
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    
    #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    for ei in input_tensor:
        encoder_output, encoder_hidden = model(ei, encoder_hidden, batch_size, encoding=True)
        
    decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    decoder_hidden = tuple(map(lambda f: f.view(2, 2, batch_size, 300)[:, 0, :, :].contiguous(), encoder_hidden))
    
    decoder_predict = []
    
    for _ in range(TARGET_LEN):
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False)
        topv, topi = decoder_output.data.topk(1)
        decoder_input = topi.detach().to(device)

        decoder_predict.append(topi.cpu().numpy())
        #loss += criterion(decoder_output, di.view(-1))

    return np.hstack(decoder_predict)

# train model
from _utils import timeSince, print_words
import time, random
import numpy as np
start = time.time()
epochs = 100
start_epoch = 0
patience = 1000
cnt = 0
best_loss = np.inf

print('start training!')
for epoch in range(start_epoch, epochs):
    total_loss = 0
    tot = len(train_loader)
    for i, (x, y) in enumerate(train_loader):
        loss = train(x, y)
        total_loss += loss
        print('epoch {}: {}/{}, loss={}'.format(epoch + 1, i, tot, loss), end='\r')
    print('epoch {}/{}: avg train loss={}'.format(epoch + 1, epochs, total_loss / tot))

    total_loss = 0  # Reset every epoch
    tot = len(valid_loader)
    for i, (x, y) in enumerate(valid_loader):
        loss = evaluate(x, y)
        total_loss += loss
        print('epoch {}: {}/{}, loss={}'.format(epoch + 1, i, tot, loss), end='\r')
    print('epoch {}/{}: avg valid loss={}'.format(epoch + 1, epochs, total_loss / tot))
    print(timeSince(start, epoch + 1, epochs))

    if total_loss / tot < best_loss:
        print('update best loss: {} -> {}, saving model...'.format(best_loss, total_loss / tot))
        best_loss = total_loss / tot
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion': criterion,
            'best_loss': best_loss
        }, CKPT_VALID_NAME)
        cnt = 0
    else:
        cnt += 1

    if (epoch + 1) % 1 == 0:
        tX, tY = random.choice(list(zip(train_X, train_Y)))
        pt = predict(torch.from_numpy(tX).view(1, -1))
        print_words('train', tX.reshape(1, -1), tY.reshape(1, -1), pt, word2vec)

        vX, vY = random.choice(list(zip(valid_X, valid_Y)))
        pv = predict(torch.from_numpy(vX).view(1, -1))
        print_words('valid', vX.reshape(1, -1), vY.reshape(1, -1), pv, word2vec)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion': criterion
        }, CKPT_NAME)

    if cnt > patience:
        print('done training')
        break
