import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def preprocessing(str):
    return re.sub("[0-9]+", "0", str)

df_train = pd.read_csv("train.txt", header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df_valid = pd.read_csv("valid.txt", header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df_test = pd.read_csv("test.txt", header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df_train = df_train.applymap(preprocessing)
df_valid = df_valid.applymap(preprocessing)
df_test = df_test.applymap(preprocessing)

tfidfvectizer = TfidfVectorizer(min_df=0.001)

df_train_feature = tfidfvectizer.fit_transform(df_train["TITLE"]).toarray()
df_valid_feature = tfidfvectizer.transform(df_train["TITLE"]).toarray()
df_test_feature = tfidfvectizer.transform(df_train["TITLE"]).toarray()

df_train_feature = pd.DataFrame(data=df_train_feature, columns=tfidfvectizer.get_feature_names_out())
df_valid_feature = pd.DataFrame(data=df_valid_feature, columns=tfidfvectizer.get_feature_names_out())
df_test_feature = pd.DataFrame(data=df_test_feature, columns=tfidfvectizer.get_feature_names_out())

df_train_feature.to_csv("train.feature.txt", sep="\t", index=False)
df_valid_feature.to_csv("valid.feature.txt", sep="\t", index=False)
df_test_feature.to_csv("test.feature.txt", sep="\t", index=False)

