import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, n_layers, direction, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.direction = direction
        
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=True if direction == 2 else False)
        
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        return output, hidden
    
    def initHidden(self, batch_size):
        h0 = torch.zeros(self.n_layers * self.direction, batch_size, self.hidden_size)
        #c0 = torch.zeros(self.n_layers * self.direction, batch_size, self.hidden_size * self.amp).to(device)
        #nn.init.xavier_uniform_(h0, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(c0, gain=nn.init.calculate_gain('relu'))
        return h0#(h0, c0)

class Attention(nn.Module):
    
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.w1 = nn.Linear(enc_hidden_size, dec_hidden_size)
        self.w2 = nn.Linear(dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1)
    
    def forward(self, hidden, enc_outputs):
        # hidden: (num_layers * num_directions, batch, hidden_size)
        # enc_outputs: (seq_len, batch, hidden_size * num_directions)
        seq_len = enc_outputs.size(0)
        batch_size = enc_outputs.size(1)
        
        score = torch.tanh(self.w1(enc_outputs.transpose(0, 1)) + self.w2(hidden.transpose(0, 1).reshape(batch_size, -1).unsqueeze(1)))
        attn_weight = torch.softmax(self.v(score), dim=1)
        context_vec = torch.sum(attn_weight * enc_outputs.transpose(0, 1), dim=1)
        
        return attn_weight, context_vec
    
class DecoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, attn_size, n_layers, direction, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.direction = direction
        
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=True if direction == 2 else False)
        self.out = nn.Linear(hidden_size * direction + attn_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden, context_vec):
        output, hidden = self.rnn(input, hidden)
        output = torch.cat((output[0], context_vec), dim=1)
        output = self.softmax(self.out(output))
        return output, hidden

class Seq2seq(nn.Module):
    
    def __init__(self, embedding, encoder, decoder, attention, dropout):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.embedding.weight.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
        
    def forward(self, input, hidden, batch_size, encoding, enc_outputs):
        if encoding:
            input = self.embedding(input).view(input.size(0), batch_size, -1)
            output, hidden = self.encoder(input, hidden)
            return output, hidden
        else:
            input = self.dropout(self.embedding(input).view(1, batch_size, -1))
            _, context_vec = self.attention(hidden, enc_outputs) # (1, batch, hidden_size * num_layers * num_directions)
            #concat_input = torch.cat((input, context_vec), dim=2)
            output, hidden = self.decoder(input, hidden, context_vec)
            return output, hidden
