import json
from itertools import chain
from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_numeric, strip_punctuation, strip_short, strip_tags, \
    strip_non_alphanum

contractions = { "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "must've": "must have", "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "that'd": "that would", "that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are" }

def normalize(s):
    s = s.lower()
    
    for k, v in contractions.items():
        s.replace(k, v)
    
        
    return strip_multiple_whitespaces(strip_non_alphanum(strip_numeric(remove_stopwords(strip_punctuation(strip_tags(s)))))).split()

def read_jsonl(path, is_train=True, return_id=False):
    X, Y, idx = [], [], []
    with open(path, 'r') as f:
        for line in f:
            js = json.loads(line)
            x, y = js['text'].strip().split('\n'), []
            if is_train:
                tar_id = js['extractive_summary']
            for i, _ in enumerate(x):
                x[i] = normalize(x[i]) + ['<EOS>']
                if is_train:
                    if i == tar_id:
                        y.append([0 for _ in range(len(x[i]) - 1)] + [1])
                    else:
                        y.append([0 for _ in range(len(x[i]))])
            
            x = list(chain.from_iterable(x))
            y = list(chain.from_iterable(y))
            X.append(x)
            Y.append(y)
            idx.append(js['id'])
    if return_id:
        return X, Y, idx
    else:
        return X, Y

import time, math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, epoch, epochs):
    now = time.time()
    s = now - since
    rs = s / epoch * (epochs - epoch)
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import numpy as np

def postprocess(X, Y, EOS_token):
    num = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    prediction = []
    ccnt = 0
    for i in range(num):
        possible = []
        cnt = 0
        for x, y in zip(X[i], Y[i]):
            if x == EOS_token:
                possible.append((cnt, 1 / (1 + np.exp(-y))))
                cnt += 1
        possible = sorted(possible, key=lambda x: x[1], reverse=True)
        final = []
        ii = 0
        for idx, p in possible:
            #if p < 0.5:
            #    break
            ii += 1
            final.append(idx)
            if ii > 1:
                break
        if len(final) == 0:
            print(possible)
            final.append(possible[0][0])
            ccnt += 1
        prediction.append(final)
    #print(ccnt)
    return prediction

def write_prediction(fp, prediction, ids):
    with open(fp, 'w') as f:
        for y, idx in zip(prediction, ids):
            mp = {"id":str(idx), "predict_sentence_index":y}
            s = json.dumps(mp)
            print(s, file=f)
