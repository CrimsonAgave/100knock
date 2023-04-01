# Q64
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，そのベクトルと類似度が最も高い単語と，その類似度を求める

import gensim
import re


vectors = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

read_f = open("questions-words.txt", "r")
writing_f = open("questions-words-result.txt", "w")

for line in read_f:
    if(line[0] == ":"):
        writing_f.write(": " + line[2:])
    else:
        words = re.split("\s", line)[:4]
        word, cos_sim = vectors.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
        writing_f.write(" ".join(words) + " " + word + " " + str(cos_sim) + "\n")

read_f.close()
writing_f.close()

