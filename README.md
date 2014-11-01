
FyLearn
=======

FyLearn is a fuzzy machine learning library, built on top of [SciKit-Learn](http://scikit-learn.org/).

SciKit-Learn contains many common machine learning algorithms, and is a good place to start if you want to play or program anything related to machine learning in Python. FyLearn is not intended to be a replacement for SciKit-Learn (in fact FyLearn depends on SciKit-Learn), but to provide an extra set of machine learning algorithms from the fuzzy logic community.

Algorithms implemented
----------------------

 - fylearn.frr.FuzzyReductionRuleClassifier
 - fylearn.frrga.FuzzyPatternClassifierGA

Usage
-----

You can use the classifiers as any other SciKit-Learn classifier:

    import fylearn.frr as frr
    l1 = frr.FuzzyReductionRuleClassifier()
    l1.fit(X, y)
    print l1.predict([1, 2, 3, 4])

    import fylearn.fpcga as fpcga
    l2 = fpcga.FuzzyPatternClassifierGA()
    l2.fit(X, y)
    print l2.predict([1, 2, 3, 4])


About
-----

FyLearn is supposed to mean "FuzzY learning", but in Danish the word "fy" means loosely translated "for shame". It has been created by the Department of Computer Science at Sri Venkateswara University, Tirupati, INDIA by PhD student [Søren](http://www.cs.svuni.in/~sorend/) as part of his research.

