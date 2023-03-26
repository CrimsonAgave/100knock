# 61
# United_Statesとコサイン類似度が高い語を10語だけ類似度と共に表示する

import gensim
import numpy as np


filename = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

top10_cos_sim = model.most_similar("United_States", topn=10)
print(top10_cos_sim)


"""
[('Unites_States', 0.7877249121665955), ('Untied_States', 0.7541369795799255), ('United_Sates', 0.7400726079940796), ('U.S.', 0.7310774326324463), ('theUnited_States', 
0.6404393315315247), ('America', 0.6178411841392517), ('UnitedStates', 0.6167312860488892), ('Europe', 0.6132988333702087), ('countries', 0.6044804453849792), ('Canada', 0.6019068956375122)]
"""
