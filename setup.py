# -*- coding: utf-8 -*-

from __future__ import print_function

import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

__version__ = '0.0.10'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='confseq',
    version=__version__,
    author='Steve Howard',
    author_email='dev@gostevehoward.com',
    url='https://github.com/gostevehoward/confseq',
    description='Confidence sequences and uniform boundaries',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    cmake_install_dir="src/confseq",
    include_package_data = True,
    install_requires=['pybind11>=2.3', 'numpy', 'matplotlib', 'multiprocess',
                      'scipy', 'pytest', 'pandas'],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
