# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='FyLearn',
    version='0.1',
    description='Fuzzy Machine Learning Algorithms',
    author='Søren Atmakuri Davidsen',
    author_email='sorend@cs.svuni.in',
    url='https://www.github.com/sorend/fylearn',
    license='MIT',
    packages=['fylearn'],
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
)
