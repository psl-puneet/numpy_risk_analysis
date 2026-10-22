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
