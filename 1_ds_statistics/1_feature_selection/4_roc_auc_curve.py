import pandas as pd
import numpy as np

def get_boston_dataset():   
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return data, target

# %%
df, target = get_boston_dataset()
df.head()

# %%
target.head()

# %%
