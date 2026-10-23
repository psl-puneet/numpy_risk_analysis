#%%
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from filtering_methods import run_random_forest

# %%
artifacts = joblib.load("feature_selection_model.joblib")
X_train_unique = artifacts["X_train_unique"]
X_test_unique = artifacts["X_test_unique"]
y_test = artifacts["y_test"]
y_train = artifacts["y_train"]
X_train = artifacts["X_train"]
X_test = artifacts["X_test"]

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
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i,j]) > threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col

# %%
%%time
print(X_train_unique[:4])
corr_features = get_correlation(X_train_unique, 0.85)

# %%
len(corr_features)

# %%
X_train_uncorr = X_train_unique.drop(labels=corr_features, axis=1)
X_test_uncorr = X_test_unique.drop(labels=corr_features, axis=1)

# %% 
X_train_uncorr.shape, X_test_uncorr.shape

# %%
%%time 
run_random_forest(X_train_uncorr, X_test_uncorr, y_train, y_test)

# %%
%%time 
run_random_forest(X_train, X_test, y_train, y_test)

# %%
# Feature Importance use for grouping of co-related features
corrmat

# %%
# Get comparision of all features using stacking in pandas
corrdata = corrmat.abs().stack()
# sorting all values in asc
corrdata = corrdata.sort_values(ascending=False)
corrdata

# %%
# perform grouping of feautures
corrdata = corrdata[corrdata>0.85]
corrdata = corrdata[corrdata<1]
corrdata

# %%
# perform grouping
corrdata = pd.DataFrame(corrdata).reset_index()
corrdata.columns = ["Features1", "Features2", "corr_value"]
corrdata

# %%
# greuping of correlated features
# analysis what happens when we have multiple feautes
grouped_feature_list = []
correlated_groups_list = []
for feature in corrdata.Features1.unique():
    if feature not in grouped_feature_list:
        correlated_block = corrdata[corrdata.Features1 == feature]
        grouped_feature_list = grouped_feature_list + list(correlated_block.Features2.unique()) + [feature]
        correlated_groups_list.append(correlated_block)

# %%
print(correlated_groups_list)
len(correlated_groups_list)

# %%
# Feature importance based on tree based classifiers
important_features = []
for group in correlated_groups_list:
    print(group)
# %%
