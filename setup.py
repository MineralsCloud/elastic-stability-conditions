#!/usr/bin/env python

from distutils.core import setup

import setuptools

setup(
    name='escpy',
    version='0.2.3',
    packages=[
        'escpy'
    ],
    url='https://github.com/MineralsCloud/elastic-stability-conditions',
    license='MIT',
    author='Qi Zhang',
    author_email='qz2280@columbia.edu',
    maintainer='Qi Zhang',
    maintainer_email='qz2280@columbia.edu',
    description='A package that implements Born stability conditions',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.5',
    install_requires=[
        'numpy'
    ],
)
