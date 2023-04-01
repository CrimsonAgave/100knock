# Q65
# 意味的アナロジーと文法的アナロジーの正解率を測定する
# gram から始まるものを文法的なアナロジー、そうでないものを意味的アナロジーの計算に用いrう

import re
import numpy as np

with open("questions-words-result.txt", "r") as f:
    data = f.read()
    split_data = re.split(">>> ", data)[1:]

for d in split_data:
    if(re.match("^gram", d)):
        lines = d.split("\n")[1:-2]
        lines = [l.split(" ") for l in lines]
        semantic_analogy = np.mean([(l[3] == l[4]) for l in lines])
    else:
        lines = d.split("\n")[1:-2]
        lines = [l.split(" ") for l in lines]
        syntactic_analogy = np.mean([(l[3] == l[4]) for l in lines])

print("ac_semantics: ", semantic_analogy)
print("ac_syntacs: ", syntactic_analogy)


"""
ac_semantics:  0.6789413118527042
ac_syntacs:  0.8455445544554455
"""