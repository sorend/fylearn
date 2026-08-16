
import numpy as np
import pytest
from sklearn.datasets import load_iris

import fylearn.fpcga as fpcga

# def test_classifier():

#     l = fpcga.FuzzyPatternClassifierLGA(iterations=8, epsilon=None)

#     X = np.array([
#         [0.1, 0.2, 0.4],
#         [0.11, 0.3, 0.5],
#         [0.07, 0.18, 0.38],
#         [0.2, 0.4, 0.8],
#         [0.18, 0.42, 0.88],
#         [0.22, 0.38, 0.78],
#     ])

#     y = np.array([
#         1,
#         1,
#         1,
#         0,
#         0,
#         0,
#     ])

#     l.fit(X, y)

#     print("protos_", l.protos_)

#     y_pred = l.predict([[0.0, 0.3, 0.35],
#                         [0.1, 0.4, 0.78]])

#     print("y_pred", y_pred)

#     assert len(y_pred) == 2
#     assert y_pred[0] == 1
#     assert y_pred[1] == 0


def test_classifier_iris():

    iris = load_iris()

    X = iris.data
    y = iris.target

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    l = fpcga.FuzzyPatternClassifierGA(iterations=100, random_state=1)

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(l, X, y, cv=10)

    assert len(scores) == 10
    assert np.mean(scores) > 0.6
    mean = np.mean(scores)

    print("mean", mean)

    assert 0.92 == pytest.approx(mean, 0.1)


def test_set_params_works():
    # regression: set_params used to call self.setattr which does not exist
    l = fpcga.FuzzyPatternClassifierGA()
    out = l.set_params(iterations=3, epsilon=None)
    assert out is l
    assert l.iterations == 3
    assert l.epsilon is None


def test_se_set_params_works():
    # regression: same bug existed for SEFuzzyPatternClassifier
    l = fpcga.SEFuzzyPatternClassifier()
    out = l.set_params(iterations=2, adjust_center=True)
    assert out is l
    assert l.iterations == 2
    assert l.adjust_center is True


def test_build_pi_membership():
    m = fpcga.build_pi_membership(np.array([0.1, 0.5, 0.9]), 0)
    assert isinstance(m, fpcga.fl.PiSet)
    # parameters must be sorted
    assert m.a <= m.r <= m.b


def test_build_trapezoidal_membership():
    m = fpcga.build_trapezoidal_membership(np.array([0.8, 0.2, 0.6, 0.4]), 0)
    assert isinstance(m, fpcga.fl.TrapezoidalSet)
    assert m.a <= m.b <= m.c <= m.d


def test_build_t_membership():
    m = fpcga.build_t_membership(np.array([0.9, 0.1, 0.5]), 0)
    assert isinstance(m, fpcga.fl.TriangularSet)
    assert m.a <= m.b <= m.c


def test_build_static_membership():
    m = fpcga.build_static_membership(np.array([]), 0)
    assert isinstance(m, fpcga.StaticFunction)
    assert m(None) == 0.5
    assert str(m) == "S(0.5)"


def test_build_membership_factory_selection():
    factories = (fpcga.build_pi_membership, fpcga.build_t_membership)
    m = fpcga.build_membership(factories, np.array([0.99, 0.1, 0.5, 0.9]), 0)
    assert isinstance(m, fpcga.fl.TriangularSet)
    m = fpcga.build_membership(factories, np.array([0.01, 0.1, 0.5, 0.9]), 0)
    assert isinstance(m, fpcga.fl.PiSet)


def test_global_chromosome_schema_indices_cover_each_membership_block():
    assert fpcga.chromosome_size(m=2, n_classes=2) == 17
    assert [
        fpcga.membership_gene_index(class_idx, feature_idx, 2)
        for class_idx in range(2)
        for feature_idx in range(2)
    ] == [1, 5, 9, 13]


def test_decode_rejects_legacy_oversized_chromosome():
    X = np.array([[0.1, 0.2], [0.8, 0.9]])
    y = np.array([0, 1])
    classes = np.array([0, 1])
    legacy_size = 2 + X.shape[1] * 5 * len(classes)
    with pytest.raises(ValueError, match="expected chromosome with 17 genes"):
        fpcga._decode(
            X.shape[1],
            X,
            y,
            (fpcga.DummyAggregationRuleFactory(fpcga.fl.prod),),
            (fpcga.build_pi_membership,),
            classes,
            np.zeros(legacy_size),
        )


def test_dummy_aggregation_rule_factory():
    f = fpcga.DummyAggregationRuleFactory(fpcga.fl.prod)
    assert f(None, None) is fpcga.fl.prod


def test_init_validation():
    with pytest.raises(ValueError):
        fpcga.FuzzyPatternClassifierGA(mu_factories=None)
    with pytest.raises(ValueError):
        fpcga.FuzzyPatternClassifierGA(aggregation_rules=None)
    with pytest.raises(ValueError):
        fpcga.FuzzyPatternClassifierGA(iterations=0)


def test_predict_before_fit_raises():
    l = fpcga.FuzzyPatternClassifierGA(iterations=1)
    with pytest.raises(Exception) as e:
        l.predict(np.array([[0.1, 0.2]]))
    assert "fit" in str(e.value)


def test_str_before_fit():
    l = fpcga.FuzzyPatternClassifierGA(iterations=1)
    assert str(l) == "Not trained"


def test_nan_classes_rejected():
    l = fpcga.FuzzyPatternClassifierGA(iterations=1)
    X = np.array([[0.1, 0.2], [0.2, 0.3], [0.9, 0.8]])
    y = np.array([0.0, 0.0, np.nan])
    with pytest.raises(Exception):
        l.fit(X, y)


def test_lga_smoke():
    X = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.9, 0.8], [0.85, 0.75]])
    y = np.array([1, 1, 1, 0, 0])
    l = fpcga.FuzzyPatternClassifierLGA(iterations=3, epsilon=None)
    l.fit(X, y)
    y_pred = l.predict([[0.12, 0.22], [0.88, 0.78]])
    assert len(y_pred) == 2
    assert set(l.protos_.keys()) == {0, 1}


def test_se_classifier_smoke_and_toggle():
    X = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3], [0.9, 0.8], [0.85, 0.75]])
    y = np.array([1, 1, 1, 0, 0])
    l = fpcga.SEFuzzyPatternClassifier(iterations=3, aggregation=fpcga.fl.mean)
    l.fit(X, y)
    assert set(l.protos_.keys()) == {0, 1}
    assert set(l.bases_.keys()) == {0, 1}
    y_pred = l.predict([[0.12, 0.22], [0.88, 0.78]])
    assert len(y_pred) == 2
    l.toggle_base()
    assert set(l.protos_.keys()) == {0, 1}
    l.toggle_base()
    assert not hasattr(l, "backups_")
