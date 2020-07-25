# ADL hw1

## Homework description

* [Link](https://docs.google.com/presentation/d/1omvZRbcbpo1gQ2hlktPV9gZzuAfEHcsJa18IoytgQjk/edit#slide=id.g8130877143_0_0) to the homework slide.
* [Link](https://www.youtube.com/watch?v=f1iXbnjK7pg&feature=youtu.be) to the homework video.
* Links to the data
  * On [Google Drive](https://drive.google.com/drive/folders/1L_ayPqKlm6KmimjTHvheLQgm2EZfajh4?usp=sharing)
  * On CSIE workstations (you may use `ssh` and `scp` to access them as long as you have a CSIE account)
    * linux1.csie.ntu.edu.tw:/tmp2/adl-hw1-data
    * linux5.csie.ntu.edu.tw:/tmp2/adl-hw1-data
* [Link](https://gist.github.com/adamlin120/8e5278cc840c137818146d151e7067e8) to the evaluation scripts. (Updated Mar. 21 19:55)
* [Link](https://drive.google.com/drive/folders/1OXmGfjAktsjgoAY9_MIvDVsZLlU4ebgv?usp=sharing) to sample preprocessing code.

## Execution details

### Caveats

1. Plotting part requires package `matplotlib` and `seaborn`
2. Need to `cd` in the directory to execute it

### Extractive

#### Train

1. put the data set in `data/{train.jsonl,valid.jsonl,test.jsonl}`
2. put the embedding file in `embeddings/numberbatch-en-19.08.txt`
3. run `python3.7 extractive/train.py`

#### Test

1. run `bash extractive/eval.py {testing_file_path} {output_file_path}`

#### Plot relative location figure

1. run `python3.7 extractive/draw_relative.py {valid_data_file_path} {output_pic_file_path}`

### Seq2seq

#### Train

1. put the data set in `data/{train.jsonl,valid.jsonl,test.jsonl}`
2. put the embedding file in `embeddings/numberbatch-en-19.08.txt`
3. run `python3.7 seq2seq/train.py`

#### Test

1. run `bash seq2seq/eval.py {testing_file_path} {output_file_path}`

#### Beam search performance compare

1. run `python3.7 seq2seq/beam_search_compare.py {valid_file_path}`

### Attention

#### Train

1. put the data set in `data/{train.jsonl,valid.jsonl,test.jsonl}`
2. put the embedding file in `embeddings/glove.840B.300d.txt`
3. run `python3.7 attention/train.py`

#### Test

1. run `bash attention/eval.py {testing_file_path} {output_file_path}`

#### Plot attention weights

1. run `python3.7 attention/draw_attn.py {valid_file_path} {output_pic_file_path}`

#### Beam search performance compare

1. run `python3.7 attention/beam_search_compare.py {valid_file_path}`
