#%%
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
artifacts = joblib.load("feature_selection_model.joblib")
X_train_unique = artifacts["X_train_unique"]

#%%
print(X_train_unique[:4])

# %%
%%time
corrmat = X_train_unique.corr()

# %%
plt.figure(figsize=(12,8))
sns.heatmap(corrmat)

# %%
def get_correlation(data, threshold):
    # data = training data
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.colmns)):
        for j in range(i):
            if abs(corrmat.iloc[i,j]) > threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col

# %%
corr_features = get_correlation(X_train_unique, 0.85)
corr_features

# %%
len(corr_features)

# %%
