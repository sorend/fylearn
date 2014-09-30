

datasets = (
    ("Iris", "iris.csv"),
    ("Pima Indians Diabetes", "diabetes.csv"),
    ("Wisconsin Breast Cancer", "breast-w.csv"),
    ("Bupa Liver Disorders", "bupa.csv"),
    ("Wine", "wine.csv"),
    ("Telugu Vowels", "vowel.csv"),
)

def path(dataset):
    import os.path as op
    return op.join(op.dirname(__file__), dataset[1])

def load(dataset):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    
    data = pd.read_csv(dataset) # read
    data = data.replace('?', np.nan) # change ? into nan
    data = data.dropna() # remove na columns
    X = data.ix[:,:-1].astype(float) # fixup as float type
    y = data["class"] # assume class attribute
    # scale to [0, 1]
    X = MinMaxScaler().fit_transform(X)
    # done
    return X, y
