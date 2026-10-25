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
df = pd.read_csv(file_path, nrows=20000)
print(df.head())

# %%
X = df.drop('TARGET', axis=1)
y = df["TARGET"]
X.shape, y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, test_size=0.2)

# %%
## Remove constants, quasi-constants, duplicate features
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X_train)
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)

# %%
X_train_T = X_train_filter.T
X_test_T = X_test_filter.T

# %%
X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)

# %%
X_train_T.duplicated().sum()

# %%
duplicated_features = X_train_T.duplicated()
features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T

# %%
X_train_unique.shape, X_test_unique.shape

# %%
# Calcualte the MI
mi = mutual_info_classif(X_train_unique, y_train)
len(mi)

# %%
mi

# %%
# this is independent of other features will not work in realworld
mi = pd.Series(mi)
mi.index = X_train_unique.columns
mi.sort_values(ascending=False, inplace=True)
mi

# %%
mi.plot.bar(figsize=(16,5))

# %%
sel = SelectPercentile(mutual_info_classif, percentile=10).fit(X_train_unique, y_train)
X_train_unique.columns[sel.get_support()]

# %%
len(X_train_unique.columns[sel.get_support()])

# %%
# Classify based on the top 23 features
sel.scores_

# %%
X_train_mi = sel.transform(X_train_unique)
X_test_mi = sel.transform(X_test_unique)
X_train_mi.shape, X_test_mi.shape

# %%
# Build the model on top k 
def run_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy on test:")
    print(accuracy_score(y_test, y_pred))

# %%
%%time
run_random_forest(X_train_mi, X_test_mi, y_train, y_test)

# %%
%%time
run_random_forest(X_train, X_test, y_train, y_test)

# %%
(811-666)*100/811

# %%
# Mutual information gain in regression
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

boston = fetch_california_housing(as_frame=True)
print(boston.DESCR)

# %%
X=pd.DataFrame(data=boston.data, columns=boston.feature_names)

# %%
X.head()

# %%
y=boston.target
y.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
mi = mutual_info_regression(X_train, y_train)
mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False, inplace=True)

# %%
mi

# %%
mi.plot.bar()

# %%
sel = SelectKBest(mutual_info_regression, k=4).fit(X_train, y_train)

# %%
X_train.columns[sel.get_support()]

# %%
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# %%
r2_score(y_test, y_predict)

# %%
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
rmse

# %%
np.std(y)

# %%
X_train_4 = sel.transform(X_train)
X_train_4.shape

# %% 
X_test_4 = sel.transform(X_test)
X_test_4.shape

# %%
model = LinearRegression()
model.fit(X_train_4, y_train)
y_predict = model.predict(X_test_4)
r2_score_val = r2_score(y_test, y_predict)
print("r2 Score ", r2_score_val)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("rmse ", rmse)

# %%
