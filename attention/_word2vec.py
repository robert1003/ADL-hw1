import numpy as np
import pickle

class Word2Vec:
    
    def __init__(self, embedding_file, embedding_dim, raw=True):
        self.embedding_dim = embedding_dim
        if raw:
            self.embedding_index = self._loadModel(embedding_file)
        else:
            self._loadEmbedding(embedding_file)
        
    def _loadEmbedding(self, embedding_file):
        print('Loading processed embedding...')
        with open(embedding_file, 'rb') as f:
            tmp = pickle.load(f)
            self.embedding = tmp['embedding']
            self.word2idx = tmp['word2idx']
            self.idx2word = tmp['idx2word']
        print('done')
        
    def _loadModel(self, embedding_file):
        print("Loading Model", embedding_file)
        model = {}
        miss_cnt = 0
        with open(embedding_file, 'r') as f:
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                try:
                    embedding = np.array(splitLine[1:]).astype('float32')
                    if len(embedding) != self.embedding_dim:
                        miss_cnt += 1
                        continue
                except:
                    miss_cnt += 1
                    continue
                model[word] = embedding
        print("Done.", len(model), "words loaded from", embedding_file, ",missed", miss_cnt, "words")
        return model
    
    def _countWords(self, datas):
        d = {}
        for data in datas:
            for sent in data:
                for word in sent:
                    if word in d:
                        d[word] += 1
                    else:
                        d[word] = 1
        return d
    
    def make_embedding(self, datas, threshold):
        word_counts = self._countWords(datas)
        
        idx = 0
        self.word2idx, self.idx2word = {}, {}
        for codes in ['<SOS>', '<EOS>', '<UNK>', '<PAD>']:
            self.word2idx[codes] = idx
            self.idx2word[idx] = codes
            idx += 1
        for word, count in word_counts.items():
            if count >= threshold or word in self.embedding_index:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
        self.embedding = []
        cnt = 0
        for word, i in self.word2idx.items():
            if word in self.embedding_index:
                self.embedding.append(self.embedding_index[word])
            else:
                self.embedding.append(np.array(np.random.uniform(-1.0, 1.0, self.embedding_dim)))
                cnt += 1
        self.embedding = np.vstack(self.embedding)
        
        print("Vocab size: {}, Embedding size: {}, Words not in pretrained embedding: {}".format(len(word_counts), len(self.word2idx), cnt))
        
        return self.embedding
    
    def _pad_sent(self, sent, sent_len, ):        
        if len(sent) > sent_len:
            sent = sent[:sent_len]
        else:
            for _ in range(sent_len - len(sent)):
                sent.append(self.word2idx['<PAD>'])
        return sent
    
    def sent2idx(self, sents, sent_len):
        sent_list = []
        unk_cnt = 0
        for i, sent in enumerate(sents):
            word_idx = []
            
            for word in sent:
                if word in self.word2idx.keys():
                    word_idx.append(self.word2idx[word])
                else:
                    word_idx.append(self.word2idx['<UNK>'])
                    unk_cnt += 1
            
            word_idx.append(self.word2idx['<EOS>'])
            word_idx = self._pad_sent(word_idx, sent_len)
            sent_list.append(word_idx)
            #print('#{} of sents processed'.format(i + 1), end='\r')
        print('#{} of sents processed'.format(len(sent_list)))
        print('#{} of unknown words'.format(unk_cnt))
        return np.vstack(sent_list)
    
