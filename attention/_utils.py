import json, time, math
from gensim.parsing.preprocessing import remove_stopwords, strip_multiple_whitespaces, strip_numeric, strip_punctuation, strip_short, strip_tags, \
    strip_non_alphanum

contractions = { "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "must've": "must have", "mustn't": "must not", "needn't": "need not", "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "that'd": "that would", "that's": "that is", "there'd": "there had", "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is", "who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are" }

def _normalize(s):
    s = s.lower()
    
    for k, v in contractions.items():
        s.replace(k, v)
        
    return strip_multiple_whitespaces(strip_non_alphanum(strip_numeric(strip_punctuation(strip_tags(s))))).split()

def _normalize_target(s):
    s = s.lower()
    
    for k, v in contractions.items():
        s.replace(k, v)
        
    return strip_multiple_whitespaces(strip_punctuation(strip_tags(s))).split()

def read_jsonl(path, is_train=True, return_id=False):
    X, Y, idx = [], [], []
    with open(path, 'r') as f:
        for line in f:
            js = json.loads(line)
            X.append(_normalize(js['text']))
            if is_train:
                Y.append(_normalize_target(js['summary']))
            if return_id:
                idx.append(js['id'])
    if return_id:
        return X, Y, idx
    else:
        return X, Y

def idx2words(target, word2vec):
    # assume 2D matrix
    result = []
    for i in range(target.shape[0]):
        temp = []
        for j in range(target.shape[1]):
            temp.append(word2vec.idx2word[target[i][j]])
        result.append(temp)

    return result

def _asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, epoch, epochs):
    now = time.time()
    s = now - since
    rs = s / epoch * (epochs - epoch)
    return '%s (- %s)' % (_asMinutes(s), _asMinutes(rs))

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

def trim(s, word2vec):
    s = idx2words(s, word2vec)[0]
    try:
        t = s.index('<EOS>')
    except:
        t = len(s)
    return ' '.join(s[:t])

def print_words(name, input, target, predict, f, word2vec):
    input = trim(input, word2vec)
    target = trim(target, word2vec)
    predict = trim(predict, word2vec)

    print(f'----------{name}----------\n input:::: {input}\n target:::: {target}\n predict:::: {predict}\n score:::: {scorer.score(target, predict)}', file=f)

def write_prediction(fp, prediction, ids, word2vec, remove_dup=True):
    with open(fp, 'w') as f:
        for y, idx in zip(prediction, ids):
            y = trim(y.reshape(1, -1), word2vec).replace('<PAD> ', '').replace(' <PAD>', '').replace('<UNK> ', '').replace(' <UNK>', '')
            if remove_dup:
                y = remove_dupes(y)
            mp = {"id":str(idx), "predict":y}
            s = json.dumps(mp)
            print(s, file=f) 
    
from itertools import groupby
import string

def mysplit(s, c, start):
    assert 0 < start <= c
    s = s.split()
    res = []
    res.append(' '.join(s[:start]))
    for i in range(start, len(s), c):
        res.append(' '.join(s[i:i + c]))
    return res

def remove_dupes(s, c=20):
    for j in range(c):
        for i in range(j):
            s = [k for k, v in groupby(mysplit(s, j, i + 1))]
            s = ' '.join(s)
    return s
