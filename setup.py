# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()


# with open('LICENSE.txt') as f:
#     licenses = f.read()


setup(
    name='cardi',
    version='1.0',
    description='cardinality estimation based on density estimators.',
    # classifiers=[
    #     'Development Status :: 3.0',
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3.8',
    #     'Topic :: Approximate Query Processing :: AQP :: Data Warehouse',
    # ],
    # classifiers=wtforms.fields.SelectMultipleField(
    #     description="Classifier",
    # ),
    keywords='cardinality estimation.',
    url='https://gitee.com/quincyma/card',
    author='Qingzhi Ma',
    author_email='qzma@suda.edu.cn',
    long_description=readme,
    # license=licenses,
    # packages=['dbestclient'],
    packages=find_packages(exclude=('experiments', 'tests', 'docs')),
    # entry_points={
    #     'console_scripts': ['dbestclient=dbestclient.main:main', 'dbestslave=dbestclient.main:slave', 'dbestmaster=dbestclient.main:master'],
    # },
    zip_safe=False,
    install_requires=[
        'numpy', 'sqlparse', 'pandas', 'scikit-learn', 'scipy',  'matplotlib', 'KDEpy'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
