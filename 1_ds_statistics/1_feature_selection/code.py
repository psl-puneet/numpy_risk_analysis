# %% Cell 1: Imports
import numpy as np
import pandas as pd
import requests
import os

# %% 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

# %%
print(os.getcwd())

# %%
file_path="../data/santander-train.csv"
url = "https://raw.githubusercontent.com/laxmimerit/Data-Files-for-Feature-Selection/refs/heads/master/santander-train.csv"
response = requests.get(url)
response.raise_for_status()

with open(file_path, "wb") as f:
    f.write(response.content)

# %%
df = pd.read_csv(file_path)
print(df.head())
# %%
X = df.drop('TARGET', axis=1)
y = df["TARGET"]

X.shape, y.shape
# %%
