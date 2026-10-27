# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold


def get_boston_dataset():   
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target

#  CRIM     per capita crime rate by town
#  ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  INDUS    proportion of non-retail business acres per town
#  CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  NOX      nitric oxides concentration (parts per 10 million)
#  RM       average number of rooms per dwelling
#  AGE      proportion of owner-occupied units built prior to 1940
#  DIS      weighted distances to five Boston employment centres
#  RAD      index of accessibility to radial highways
#  TAX      full-value property-tax rate per $10,000
#  PTRATIO  pupil-teacher ratio by town
#  B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#  LSTAT    % lower status of the population
#  MEDV  


# %%
df, target = get_boston_dataset()
headers = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
df = pd.DataFrame(data=df, columns=headers)
df.head()

# %%
target = pd.DataFrame(data=target, columns=['MEDV'])
target.head()

# %%
# MSE for Regression, ROC_AUC for classification
# %%
file_path="../data/santander-train.csv"

# %%
df = pd.read_csv(file_path)
print(df.head())

# %%
X = df.drop('TARGET', axis=1)
y = df['TARGET']

X.shape, y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# %% 
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(X_train)
X_train_filter = constant_filter.transform(X_train)
X_test_filter = constant_filter.transform(X_test)

# %%
X_train.shape, X_test.shape

# %%
X_train_T = X_train_filter.T
X_test_T = X_test_filter.T

X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)

# %%
X_train_T.duplicated().sum()
duplicated_features = X_train_T.duplicated()

# %% 
features_to_keep = [not index for index in duplicated_features]
X_train_unique = X_train_T[features_to_keep].T
X_test_unique = X_test_T[features_to_keep].T

# %%
X_train_unique.shape, X_test_unique.shape

# %%
def run_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) 
    print("Accuracy on test set: ")
    print("Accuracy Score: ",accuracy_score(y_test, y_pred))

# %%
%%time
run_random_forest(X_train_unique, X_test_unique, y_train, y_test)

# %%
%%time
run_random_forest(X_train_filter, X_test_filter, y_train, y_test)

# %%
%%time
run_random_forest(X_train, X_test, y_train, y_test)

# %%
