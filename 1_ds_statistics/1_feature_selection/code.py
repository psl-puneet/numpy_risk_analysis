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
### Applying feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# %%
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train)

# %%
constant_filter.get_support().sum()


# %%
constant_list = [not temp for temp in constant_filter.get_support()]
constant_list

# %% 
X.columns[constant_list]
# %%
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)

# %%
X_train_filter.shape, X_test_filter.shape, X_train.shape

### Quassi Constant feature removal
# %%
quasi_constant_filter = VarianceThreshold(threshold=0.01)
quasi_constant_filter.fit(X_train_filter)

# %%
quasi_constant_filter.get_support().sum()

# %%
X_train_quassi_filter = quasi_constant_filter.transform(X_train_filter)
X_test_quassi_filter = quasi_constant_filter.transform(X_test_filter)

# %%
X_train_quassi_filter.shape , X_test_quassi_filter.shape, X_train_filter.shape

### Duplicate Features remocal
# %%
