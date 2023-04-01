# 68
# Ward方による階層型クラスタリングを実行し、結果をデンドログラムとして表示する

import gensim
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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

k = 8
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(country_vecs)
for i in range(k):
    cluster = np.where(kmeans.labels_ == i)[0]
    print("cluster:", i)
    print(", ".join(country_names[j] for j in cluster))


tsne = TSNE(n_components=2, random_state=64)
X_reduced = tsne.fit_transform(np.array(country_vecs))
plt.figure(figsize=(10, 10))
for x, country, color in zip(X_reduced, country_names, kmeans.labels_):
    plt.text(x[0], x[1], country, color='C{}'.format(color))
plt.xlim([-12, 15])
plt.ylim([-15, 15])
plt.savefig('69.png')
plt.show()
