# 61
# United_StatesとU.S.のコサイン類似度を計算する

import gensim
import numpy as np


filename = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

vec1 = np.array(model["United_States"])
vec2 = np.array(model["U.S."])

cos_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(cos_similarity)


"""
0.7310775
"""
