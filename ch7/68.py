# 68
# Ward方による階層型クラスタリングを実行し、結果をデンドログラムとして表示する

import gensim
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

filename = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)


with open("countries.csv", "r") as f:
    data = []
    for line in f:
        if(line == "Name,Code\n"): continue
        elements = line.replace("\n", "").split(",")
        data.append(elements[0])

country_vecs = []
country_names = []
for name in data:
    try:
        country_vecs.append(model[name])
        country_names.append(name)
    except:
        pass


result = linkage(country_vecs, method="ward")
plt.figure(figsize=(15, 5))
dendrogram(result, labels=country_names)
plt.show()
plt.savefig("68.png")