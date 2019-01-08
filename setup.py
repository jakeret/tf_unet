#!/usr/bin/env python

import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requires = [] #during runtime
tests_require=['pytest>=2.3'] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='tf_unet',
    version='0.1.2',
    description='Unet TensorFlow implementation',
    long_description=readme + '\n\n' + history,
    author='Joel Akeret',
    url='https://github.com/jakeret/tf_unet',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'tf_unet': 'tf_unet'},
    include_package_data=True,
    install_requires=requires,
    license='GPLv3',
    zip_safe=False,
    keywords='tf_unet',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    tests_require=tests_require,
)
