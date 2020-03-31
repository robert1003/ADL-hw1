import json, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if len(sys.argv) != 4:
    raise Exception('usage: python3 plot_p4.py prediction_path target_path pic_path')
prediction_path = sys.argv[1]
target_path = sys.argv[2]
pic_path = sys.argv[3]

predictions = {}
with open(prediction_path, 'r') as f:
    for line in f:
        js = json.loads(line)
        predictions[js['id']] = js['predict_sentence_index']

target_len = {}
with open(target_path, 'r') as f:
    for line in f:
        js = json.loads(line)
        target_len[js['id']] = len(js['sent_bounds'])

relative_loc = []
for idx, preds in predictions.items():
    for pred in preds:
        relative_loc.append(pred / target_len[idx])

sns.distplot(relative_loc, kde=False)
plt.savefig(pic_path)
