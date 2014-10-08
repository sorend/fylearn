

datasets = (
    ("Iris", "iris.csv"),
    ("Pima Indians Diabetes", "diabetes.csv"),
    ("Wisconsin Breast Cancer", "breast-w.csv"),
    ("Bupa Liver Disorders", "bupa.csv"),
    ("Wine", "wine.csv"),
    ("Telugu Vowels", "vowel.csv"),
    ("Haberman", "haberman.csv"),
    ("Indian Liver Patient", "indian-liver.csv"),
    ("Vertebral Column", "column_2C.csv"),
    ("Glass", "glass.csv"),
    ("Ionosphere", "ionosphere.csv"),
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

def cross_val_score(l, X, y, cv=10):

    from sklearn import cross_validation
    from sklearn import metrics

    skf = cross_validation.StratifiedKFold(y, n_folds=cv)

    scores = []

    for train_idx, test_idx in skf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # fit and predict
        l.fit(X_train, y_train)
        y_pred = l.predict(X_test)

        scores.append(metrics.precision_recall_fscore_support(y_test, y_pred))

    return scores
