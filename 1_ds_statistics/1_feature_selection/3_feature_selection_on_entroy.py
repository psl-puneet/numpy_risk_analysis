# %%
# Mutual information also known as entropy
import numpy as np
import pandas as pd
import requests
import os
import joblib

# %% 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile

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

