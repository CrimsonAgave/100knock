import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


filename = "newsCorpora.csv"

df = pd.read_csv(filename, header=None, sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
df = df.loc[df["PUBLISHER"].isin(["Reuters", "Huffington Post", "Busuinessweek", "Contactmusic.com", "Daily Mail"])]

train, tmp = train_test_split(df, test_size=0.2, shuffle=True)
valid, test = train_test_split(tmp, test_size=0.2, shuffle=True)

train.to_csv("train.txt", sep="\t", index=False)
valid.to_csv("valid.txt", sep="\t", index=False)
test.to_csv("test.txt", sep="\t", index=False)
