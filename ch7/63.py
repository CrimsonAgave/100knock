# 63
# Spain の単語ベクトルから Madrid のベクトルを引き、Athens のベクトルを足したベクトルを計算し、それと類似度の高い10語を類似度とともに表示する

import gensim


filename = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

v1 = model["Spain"]
v2 = model["Madrid"]
v3 = model["Athens"]

v4 =  v1 - v2 + v3

top10_cos_sim = model.most_similar(v4, topn=10)
for k, v in top10_cos_sim:
    print(k, v)


"""
Athens 0.7528455853462219
Greece 0.6685471534729004
Aristeidis_Grigoriadis 0.5495777726173401
Ioannis_Drymonakos 0.5361456871032715
Greeks 0.5351787805557251
Ioannis_Christou 0.5330225825309753
Hrysopiyi_Devetzi 0.5088489651679993
Iraklion 0.5059264302253723
Greek 0.5040616393089294
Athens_Greece 0.503411054611206
"""
