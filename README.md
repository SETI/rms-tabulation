[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-tabulation/run-tests.yml?branch=main)](https://github.com/SETI/rms-tabulation/actions)
[![Documentation Status](https://readthedocs.org/projects/rms-tabulation/badge/?version=latest)](https://rms-tabulation.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-tabulation/main?logo=codecov)](https://codecov.io/gh/SETI/rms-tabulation)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-tabulation)](https://pypi.org/project/rms-tabulation)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-tabulation)](https://pypi.org/project/rms-tabulation)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-tabulation)](https://pypi.org/project/rms-tabulation)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-tabulation)](https://pypi.org/project/rms-tabulation)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-tabulation/latest)](https://github.com/SETI/rms-tabulation/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-tabulation)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-tabulation)](https://github.com/SETI/rms-tabulation/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-tabulation)

# Introduction

`tabulation` is a Python module that provides the `Tabulation` class. The `Tabulation`
class represents a mathematical function by a sequence of linear interpolations between
points defined by arrays of *x* and *y* coordinates.

`tabulation` is a product of the [PDS Ring-Moon Systems Node](https://pds-rings.seti.org).

# Installation

The `tabulation` module is available via the `rms-tabulation` package on PyPI and can be
installed with:

```sh
pip install rms-tabulation
```

# Getting Started

The `Tabulation` class models a mathematical function by a series of (*x*,*y*) points and
performs linear interpolation between them. The function is assumed to be zero outside of
the defined *x* domain. However, if provided, one zero-valued point at either the
beginning and/or the end of the *x* domain is considered to be valid data so that the
interpolation can be anchored. If there is no zero-valued point provided, a step function
is assumed.

A variety of mathematical operations can be performed on `Tabulation` objects, including
addition, subtracting, multiplication, division, integration, and finding the mean, FWHM,
or square width. See the
[module documentation](https://rms-.readthedocs.io/en/latest/module.html) for details.

Here are some examples to get you started:

```python
>>> from tabulation import Tabulation
>>> t1 = Tabulation([2, 4], [10, 10])  # Leading&trailing step function
>>> t1.domain()
(2., 4.)
>>> t1([0,   1,   1.9, 2,   3,   3.9, 4,   5,   6])
array([ 0.,  0.,  0., 10., 10., 10., 10.,  0.,  0.])
>>> t1.mean()
10.0

>>> t2 = Tabulation([0, 2, 4], [0, 10, 10])  # Ramp on leading edge
>>> t2.domain()
(0., 4.)
>>> t2([0,   1,   1.9,  2,   3,   3.9, 4,   5,   6])
array([ 0.,  5.,  9.5, 10., 10., 10., 10.,  0.,  0.])
>>> t2.mean()
7.5

>>> t3 = Tabulation([1, 3, 5], [5, 10, 5])  # Another step function
>>> r1 = t1 - t3
>>> r1.domain()
(1.0, 5.0)
>>> r1.x
array([ 1.,   2.,   3.,   4.,   5.])  # Now includes all x points from both t1 and t3
>>> r1.y
array([-5. ,  2.5,  0. ,  2.5, -5. ])




# Contributing

Information on contributing to this package can be found in the
[Contributing Guide](https://github.com/SETI/rms-/blob/main/CONTRIBUTING.md).

# Links

- [Documentation](https://rms-.readthedocs.io)
- [Repository](https://github.com/SETI/rms-)
- [Issue tracker](https://github.com/SETI/rms-/issues)
- [PyPi](https://pypi.org/project/rms-)

# Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-/blob/main/LICENSE).