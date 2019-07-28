# -*- coding: utf-8 -*-
"""fylearn

:author: Soren Atmakuri
:copyright: 2013-2019
:license: MIT

fylearn, a Python library for fuzzy machine learning.
"""
from setuptools import setup, find_packages
import fylearn

MY_VERSION = fylearn.__version__

setup(
    name='fylearn',
    packages=find_packages(),
    version=MY_VERSION,
    description='Fuzzy Machine Learning Algorithms',
    author='Søren Atmakuri Davidsen',
    author_email='sorend@cs.svu-ac.in',
    url='https://github.com/sorend/fylearn',
    download_url='https://github.com/sorend/fylearn/tarball/%s' % (MY_VERSION,),
    license='MIT',
    keywords=['machine learning', 'fuzzy logic', 'scikit-learn'],
    install_requires=[
        'numpy>=1.16',
        'scipy>=1.3',
        'scikit-learn>=0.20',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
