### Extractive

#### Train

1. put the data set in `data/{train.jsonl,valid.jsonl,test.jsonl}`
2. put the embedding file in `embeddings/numberbatch-en-19.08.txt`
3. run `python3 extractive/train.py`

#### Test

1. run `bash extractive/eval.py {testing_file_path} {output_file_path}`

#### Plot relative location figure

1. run `python3 extractive/draw_relative.py {valid_data_file_path} {output_pic_file_path}`

### Seq2seq

#### Train

1. put the data set in `data/{train.jsonl,valid.jsonl,test.jsonl}`
2. put the embedding file in `embeddings/numberbatch-en-19.08.txt`
3. run `python3 seq2seq/train.py`

#### Test

1. run `bash seq2seq/eval.py {testing_file_path} {output_file_path}`

#### Beam search performance compare

1. run `python3 seq2seq/beam_search_compare.py {valid_file_path}`

### Attention

#### Train

1. put the data set in `data/{train.jsonl,valid.jsonl,test.jsonl}`
2. put the embedding file in `embeddings/glove.840B.300d.txt`
3. run `python3 attention/train.py`

#### Test

1. run `bash attention/eval.py {testing_file_path} {output_file_path}`

#### Plot attention weights

1. run `python3 attention/draw_attn.py {valid_file_path} {output_pic_file_path}`

#### Beam search performance compare

1. run `python3 attention/beam_search_compare.py {valid_file_path}`
