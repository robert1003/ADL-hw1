import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, amp, n_layers, direction, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.direction = direction
        self.amp = amp
        
        self.rnn = nn.GRU(hidden_size, hidden_size * amp, num_layers=n_layers, dropout=dropout, bidirectional=True if direction == 2 else False)
        
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        return output, hidden
    
    def initHidden(self, batch_size):
        h0 = torch.ones(self.n_layers * self.direction, batch_size, self.hidden_size * self.amp)
        #h0 = nn.init.xavier_uniform_(torch.zeros(self.n_layers * self.direction, batch_size, self.hidden_size * self.amp))
        return h0
    
class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size, amp, n_layers, direction, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.direction = direction
        self.amp = amp
        
        self.rnn = nn.GRU(hidden_size, hidden_size * amp, num_layers=n_layers, dropout=dropout, bidirectional=True if direction == 2 else False)
        self.out = nn.Linear(hidden_size * amp * direction, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Seq2seq(nn.Module):
    
    def __init__(self, embedding, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.embedding.weight.requires_grad = True
        
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
        
    def forward(self, input, hidden, batch_size, encoding):
        if encoding:
            output = self.embedding(input).view(1, batch_size, -1)
            output, hidden = self.encoder(output, hidden)
            return output, hidden
        else:
            output = torch.tanh(self.embedding(input).view(1, batch_size, -1))
            output, hidden = self.decoder(output, hidden)
            return output, hidden
