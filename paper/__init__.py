
import fylearn.fpcga as fpcga
import fylearn.frr as frr
from sklearn import tree, svm, neighbors
import fylearn.fuzzylogic as fl

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
    ("Balance Scale", "balance-scale.csv"),
)

learners = (
    ("CART", tree.DecisionTreeClassifier()),
    ("SVM", svm.SVC(kernel='linear')),
    ("kNN", neighbors.KNeighborsClassifier()),
    (r"$\text{FRR}_{\text{prod}}$", frr.FuzzyReductionRuleClassifier(aggregation=fl.prod)),
    (r"$\text{FRR}_{\text{mean}}$", frr.FuzzyReductionRuleClassifier(aggregation=fl.mean)),
    (r"$\text{FPC}_{\text{pp}}$", fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.prod,), iterations=100, epsilon=None)),
    (r"$\text{FPC}_{\text{pm}}$", fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.mean,), iterations=100, epsilon=None)),
    (r"$\text{FPC}_{\text{tp}}$", fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_t_membership,), aggregation_rules=(fl.prod,), iterations=100, epsilon=None)),
    (r"$\text{SFPC_{\text{pp}}}$", fpcga.FuzzyPatternClassifierGA2(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.prod,), iterations=25, epsilon=None)),
    (r"$\text{SFPC}_{\text{pm}}$", fpcga.FuzzyPatternClassifierGA2(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.mean,), iterations=25, epsilon=None)),
    # fpcga.FuzzyPatternClassifierGA(iterations=100, epsilon=None), # all
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
    y = data["class"].astype(basestring) # assume class attribute
    # scale to [0, 1]
    X = MinMaxScaler().fit_transform(X)
    # done
    return X, y

def cross_val_score(l, X, y, cv=10):

    from sklearn import cross_validation
    from sklearn import metrics
    import numpy as np
    from sklearn.utils import check_arrays, column_or_1d

    skf = cross_validation.StratifiedKFold(y, n_folds=cv)
    X, = check_arrays(X)
    y, = check_arrays(y)

    test_scores = []
    training_scores = []

    for train_idx, test_idx in skf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        y_train = y_train.astype(str)
        y_test  = y_test.astype(str)

        # fit and predict
        l.fit(X_train, y_train)

        # training scores
        y_pred = l.predict(X_train)

        training_scores.append(metrics.accuracy_score(y_train, y_pred))

        # testing scores
        y_pred = l.predict(X_test)
        test_scores.append(metrics.accuracy_score(y_test, y_pred))

    return (test_scores, training_scores)
