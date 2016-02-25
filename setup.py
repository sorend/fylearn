# -*- coding: utf-8 -*-
from setuptools import setup

MY_VERSION = '0.1.2'

setup(
    name='fylearn',
    packages=['fylearn'],
    version=MY_VERSION,
    description='Fuzzy Machine Learning Algorithms',
    author='Søren Atmakuri Davidsen',
    author_email='sorend@cs.svuni.in',
    url='https://github.com/sorend/fylearn',
    download_url='https://github.com/sorend/fylearn/tarball/%s' % (MY_VERSION,),
    license='MIT',
    keywords=['machine learning', 'fuzzy logic', 'scikit-learn'],
    install_requires=[
        'numpy>=1.9',
        'scipy>=0.16',
        'scikit-learn>=0.16',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
