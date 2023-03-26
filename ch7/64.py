# 64
# 単語アナロジーの評価データをダウンロードし、vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算してそのベクトルが最も高い単語とその類似度を求め、各事例の末尾に追記する

import gensim


# データの入力
filename1 = "GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(filename1, binary=True)

filename2 = "questions-words.txt"
with open(filename2) as f:
    data = f.readlines()

    # 単語アナロジー
    words = []
    for l in data:
        if not(":" in l):
            words.append(l.replace("\n", "").split(" "))
        else:
            words.append(l)
            
    for i in range(len(words)):
        if(words[i][0] == ":"):
            pass
        else:
            v1 = model[words[i][0]]
            v2 = model[words[i][1]]
            v3 = model[words[i][2]]
            
            v4 = v2 - v1 + v3
            v_sim = model.most_similar(v4, topn=1)

            words[i].append(v_sim[0][0])
            words[i].append(v_sim[0][1])

        
    print("end")
    f.write(words)

