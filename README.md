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
performs linear interpolation between them. Although optimized to model filter bandpasses
and spectral flux, the class is sufficiently general to be used in a wide range of
applications.

The mathematical function is treated as equal to zero outside the domain of the *x*
coordinates, with a step at the provided leading and trailing *x* coordinates. In general,
zero values (either supplied or computed) at either the leading or trailing ends are
removed. However, if explicitly supplied, one leading and/or trailing zero value is
considered significant because it anchors the interpolation of a ramp at the beginning or
end of the domain.

A variety of mathematical operations can be performed on `Tabulation` objects, including
addition, subtraction, multiplication, division, integration, and finding the X mean,
FWHM, and square width. See the [module
documentation](https://rms-.readthedocs.io/en/latest/module.html) for details.

Here are some examples to get you started:

```python
>>> from tabulation import Tabulation
>>> t1 = Tabulation([2, 4], [10, 10])  # Leading&trailing step function
>>> t1.domain()
(2., 4.)
>>> r1 = t1([0,   1,   1.9, 2,   3,   3.9, 4,   5,   6])
array([      0.,  0.,  0., 10., 10., 10., 10.,  0.,  0.])
>>> t1.x_mean()
3.0
>>> t1.integral()
20.0

>>> t2 = Tabulation([0, 2, 4], [0, 5, 5])  # Ramp on leading edge
>>> t2.domain()
(0., 4.)
>>> r2 = t2([0,   1,   1.9,  2,   3,   3.9, 4,   5,   6])
array([      0.,  5.,  9.5, 10., 10., 10., 10.,  0.,  0.])
>>> t2.x_mean()
2.6666666666666665
>>> t2.integral()
15.0

>>> t3 = t1-t2
>>> t3.domain()
(0.0, 4.0)
>>> r2-r1
array([ 0.  ,  2.5 ,  4.75, -5.  , -5.  , -5.  , -5.  ,  0.  ,  0.  ])
>>> t3([0,       1,   1.9,   2,     3,     3.9,   4,     5,     6])
array([ 0.  ,  2.5 ,  4.75, -5.  , -5.  , -5.  , -5.  ,  0.  ,  0.  ])
>>> t3.integral()
5.000000000000001
```

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