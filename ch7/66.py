# Q 66
# WordSimilarity-353の評価データをダウンロードし、
# 単語ベクトルにより計算される類似度のランキングと、人間の類似度判定のランキングの間の
# スピアマン相関係数を計算する

import gensim
import numpy as np

filename = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

with open("./wordsim353/combined.csv") as f:
    human_eval = []
    machine_eval = []
    for line in f:
        if("Word 1,Word 2,Human (mean)\n" in line): continue
        elements = line.replace("\n", "").split(",")
        human_eval.append(float(elements[2]))
        machine_eval.append(model.similarity(elements[0], elements[1]))

human_eval = np.array(human_eval)
machine_eval = np.array(machine_eval)
d = human_eval - machine_eval
n = len(human_eval)
spearman_coef = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1)) 

print(spearman_coef)


"""
0.9982972326270444
"""