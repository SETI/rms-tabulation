##########################################################################################
# tabulation/__init__.py
##########################################################################################
"""
Tabulation class,
PDS Ring-Moon Systems Node, SETI Institute

The Tabulation class represents a mathematical function by a sequence of linear
interpolations between points defined by arrays of x and y coordinates. Although optimized
to model filter bandpasses and spectral flux, the class is sufficiently general to be used
in a wide range of applications. See the documentation for the Tabulation class for full
details.
"""

import math
import numbers
import numpy as np
from scipy.interpolate import interp1d

try:
    from math import nextafter as _nextafter    # Only in Python 3.9 and later
except ImportError:                             # pragma: no cover
    from numpy import nextafter as _nextafter

# We use the `steps` option only implemented in Python 3.12. Sheesh. Here's a workaround.
nextafter = _nextafter
try:
    x = nextafter(1, math.inf, steps=2)
except TypeError:                               # pragma: no cover
    def nextafter(x, y, /, *, steps=1):
        for i in range(steps):
            x = _nextafter(x, y)
        return x

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = 'Version unspecified'


class Tabulation(object):
    """A class that represents a function by a sequence of linear interpolations.

    Although optimized to model filter bandpasses and spectral flux, the class is
    sufficiently general to be used in a wide range of applications.

    The interpolations are defined between points defined by arrays of x and y
    coordinates. The mathematical function is treated as equal to zero outside the domain
    of the x coordinates, with a step at the provided leading and trailing x coordinates.

    The internal arrays of a Tabulation can be accessed directly via the `x` and `y`
    attributes. However, these arrays are not writeable.

    Tabulation arithmetic is supported, using the standard `+`. `-`, `*`, and `/`
    operators. In-place operators `+=`, `-=`, `*=`, and `/=` are also supported. A
    Tabulation can be "sliced" using standard NumPy index notation; for example, `t[:10]`
    is a new Tabulation containing the first ten elements of Tabulation `t`.

    In general, zero values (either supplied or computed) at either the leading or
    trailing ends are removed. However, if explicitly supplied, one leading and/or
    trailing zero value is considered significant because it anchors the interpolation of
    a ramp at the beginning or end of the domain. For example::

        >>> t1 = Tabulation([2, 4], [10, 10])  # Leading & trailing step function
        >>> t1.domain()
        (2., 4.)
        >>> t1([0,   1,   1.9, 2,   3,   3.9, 4,   5,   6])
        array([ 0.,  0.,  0., 10., 10., 10., 10.,  0.,  0.])

        >>> t2 = Tabulation([0, 2, 4], [0, 5, 5])  # Ramp on leading edge
        >>> t2.domain()
        (0., 4.)
        >>> t2([0,    1,    1.9,  2,    3,    3.9,  4,    5,    6])
        array([ 0.  , 2.5 , 4.75, 5.  , 5.  , 5.  , 5.  , 0.  , 0.  ])

    By default it is assumed that the function never has leading or trailing zeros beyond
    the single zero necessary to anchor the interpolation, and the Tabulation object will
    automatically trim any additional leading and/or trailing regions of the domain that
    have purely zero values.

    When mathematical operations are performed on Tabulations, new x-coordinates are added
    as necessary to keep the behavior of step functions. For example::

        >>> t1.x
        array([2., 4.])
        >>> t2.x
        array([0., 2., 4.])
        >>> (t1-t2).x
        array([0., 2., 2., 4.])
        >>> (t1-t2).y
        array([ 0., -5.,  5.,  5.])

    Note that the new x-coordinates are epsilon away from the adjacent x-coordinates,
    essentially producing an infinitesimally narrow ramp to simulate the original step
    function::

        >>> (t1-t2).x[1]
        1.9999999999999998
        >>> (t1-t2).x[2]
        2.0
    """

    __array_priority__ = 1      # Enable "constant * Tabulation" operation to work

    def __init__(self, x, y):
        """Constructor for a Tabulation object.

        Parameters:
            x (array-like): A 1-D array of x-coordinates, which must be monotonic (either
                increasing or decreasing).
            y (array-like): A 1-D array of y-values, given in the same order as the
                x-coordinates.
        """

        self._update(x, y)

    @property
    def shape(self):
        """The shape of this Tabulation as a tuple."""

        return (self._length,)

    ########################################
    # Private methods
    ########################################

    def _update(self, x, y):
        """Update a Tabulation in place with new x and y arrays. Trim the result.

        Parameters:
            x (array-like): The new 1-D array of x-coordinates; must be monotonic.
            y (array-like): The new 1-D array of y-coordinates.

        Returns:
            Tabulation: The current Tabulation object mutated with the new arrays.

        Raises:
            ValueError: If the x and/or y arrays do not have the proper dimensions,
                size, or monotonicity.
        """

        xx = np.asarray(x, dtype=np.float64)    # makes a copy only if necessary
        x_copied = xx is not x
        x = xx

        yy = np.asarray(y, dtype=np.float64)
        y_copied = yy is not y
        y = yy

        if x.ndim != 1:
            raise ValueError('x array is not 1-dimensional')

        if x.shape != y.shape:
            raise ValueError('x and y arrays do not have the same size')

        if not np.all(np.isfinite(x)):
            raise ValueError('x array cannot contain NaNs or infinities')

        if not np.all(np.isfinite(y)):
            raise ValueError('y array cannot contain NaNs or infinities')

        if not x.size:
            return self._update([0.], [0.])

        # Swap x-coordinates to increasing
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]

        # Trim...
        x_before = []
        y_before = []
        x_after = []
        y_after = []

        nonzeros = np.where(y)[0]
        if nonzeros.size:

            # Slice out the endpoints and their adjacent zeros
            # Ignore "ramp" x-values, falling within 3 * epsilon of the adjacent x
            first = nonzeros[0]
            if first > 0 and x[first-1] < nextafter(x[first], -math.inf, steps=3):
                first -= 1
            else:
                x_before = [nextafter(x[first], -math.inf)]
                y_before = [0.]

            last = nonzeros[-1]
            if last < x.size-1 and x[last+1] > nextafter(x[last], math.inf, steps=3):
                last += 1
            else:
                x_after = [nextafter(x[last], math.inf)]
                y_after = [0.]

            x = x[first:last+1]
            y = y[first:last+1]

            # Save the length and domain limits before dealing with new ramps
            self._xmin = x[0]
            self._xmax = x[-1]
            self._length = x.size

            # Insert the ramps as needed
            if x_before or x_after:
                x = np.concatenate((x_before, x, x_after))
                y = np.concatenate((y_before, y, y_after))
                x_copied = True
                y_copied = True

        else:
            # If the values are all zero, retain original domain and sampling
            self._xmin = x[0]
            self._xmax = x[-1]
            self._length = x.size
            x_before = []

        # Make sure the sequence is monotonic but tolerate duplicates for now. This is
        # necessary because if x contains an infinitesimal step, an operation on x
        # might result in a duplicate.
        if not np.all(x[:-1] <= x[1:]):
            raise ValueError('x-coordinates are not strictly monotonic')

        # Separate duplicated x by epsilon, shifting the one with y closer to zero
        dups = np.where(x[:-1] == x[1:])[0]
        if dups.size:
            if not x_copied:
                x = x.copy()
                x_copied = True

            for i in dups:
                if abs(y[i]) < abs(y[i+1]):
                    x[i] = nextafter(x[i], -math.inf)
                else:
                    x[i+1] = nextafter(x[i], math.inf)

        # Fill in the arrays, with _x and _y including the ramp endpoints
        self._x = x if x_copied else x.copy()
        self._y = y if y_copied else y.copy()
        self._x.flags.writeable = False
        self._y.flags.writeable = False

        i0 = 1 if x_before else 0
        self.x = self._x[i0:i0+self._length]
        self.y = self._y[i0:i0+self._length]

        self._func = None       # filled in if/when needed
        return self

    @staticmethod
    def _xmerge(x1, x2):
        """The union of x-coordinates found in each of the given arrays.

        Parameters:
            x1 (array-like): The first array of x-coordinates.
            x2 (array-like): The second array of x-coordinates.

        Returns:
            np.array: The merged array of x-coordinates.
        """

        return np.sort(np.unique(np.hstack((x1, x2))))

    def _xoverlap(self, other):
        """The union of x-coords that fall within the intersection of two Tabulations.

        Parameters:
            other (Tabulation or array-like): The second Tabulation or an array of
                monotonically increasing x-coordinates.

        Returns:
            np.array: The merged array of x-coordinates, limited to those values that
            fall within the intersection of the domains of the two arrays.

        Raises:
            ValueError: If the domains do not overlap.
        """

        if isinstance(other, Tabulation):
            x2 = other._x
            xmin = other._xmin
            xmax = other._xmax
        else:
            x2 = other
            xmin = x2[0]
            xmax = x2[1]

        xmin = max(self._xmin, xmin)
        xmax = min(self._xmax, xmax)
        if xmin > xmax:
            raise ValueError('domains do not overlap')

        x1 = self._x
        x1 = x1[(x1 >= xmin) & (x1 <= xmax)]
        x2 = x2[(x2 >= xmin) & (x2 <= xmax)]
        return Tabulation._xmerge(x1, x2)

    ########################################
    # Standard operators
    ########################################

    def __call__(self, x):
        """The interpolated value corresponding to an x-coordinate.

        This definition allows any Tabulation to be treated as a function, so if `tab` is
        a Tabulation, `tab(x)` returns the value of that Tabulation evaluated at `x`.

        Parameters:
            x (float or array-like): The x-coordinate(s) at which to evaluate the
                Tabulation.

        Returns:
            float or array-like: The value(s) of the interpolated y-coordinate at the
            given x-coordinate(s).
        """

        # Fill in the 1-D interpolation if necessary
        if self._func is None:
            self._func = interp1d(self.x, self.y, kind='linear',
                                  bounds_error=False, fill_value=0.)

        value = self._func(x)
        if np.shape(x):
            return value

        return float(value[()])

    def __mul__(self, other):
        """Multiply, supporting the "`*`" operator.

        Parameters:
            other (Tabulation, float, or int): The multiplier.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            TypeError: If a Tabulation cannot be multiplied by an object of the given
                type.
            ValueError: If the domains of the two Tabulations do not overlap.

        Notes:
            If `other` is a Tabulation, the new domain is the intersection of the domains
            of the two Tabulations. Within this domain, the product is evaluated at the
            union of the Tabulations' x-coordinates. Because interpolation is linear,
            values between the new x-coordinates will have reduced accuracy; if this is a
            concern, subsample one of the Tabulations first.
        """

        if isinstance(other, Tabulation):
            new_x = self._xoverlap(other)
            return Tabulation(new_x, self(new_x) * other(new_x))

        # Explicitly disallow multiplication by an array; this might work by accident
        if isinstance(other, np.ndarray):
            raise TypeError("unsupported operand type(s) for *: 'Tabulation' and "
                            "'numpy.ndarray'")

        # Raises TypeError on incompatible type
        return Tabulation(self._x, self._y * other)

    def __rmul__(self, other):
        """Multiply, supporting the "`*`" operator if the Tabulation comes second.

        Parameters:
            other (float or int): The multiplier.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            TypeError: If a Tabulation cannot be multiplied by an object of the given
                type.
        """

        return self.__mul__(other)

    def __imul__(self, other):
        """In-place multiply, supporting the "`*=`" operator.

        Parameters:
            other (Tabulation, float, or int): The multiplier.

        Returns:
            Tabulation: The current Tabulation, mutated with the new values.

        Raises:
            TypeError: If the Tabulation cannot be multiplied by an object of the given
                type.
            ValueError: If the domains of the two Tabulations do not overlap.

        Notes:
            If `other` is a Tabulation, the new domain is the intersection of the domains
            of the two Tabulations. Within this domain, the product is evaluated at the
            union of the Tabulations' x-coordinates.

            Because interpolation is linear, values between the new x-coordinates will
            have reduced accuracy; if this is a concern, subsample one of the Tabulations
            first.
        """

        if isinstance(other, Tabulation):
            new_x = self._xoverlap(other)
            return self._update(new_x, self(new_x) * other(new_x))

        # Explicitly disallow multiplication by an array; this might work by accident
        if isinstance(other, np.ndarray):
            raise TypeError("unsupported operand type(s) for *=: 'Tabulation' and "
                            "'numpy.ndarray'")

        # Raises TypeError on incompatible type
        return self._update(self._x, self._y * other)

    def _divide(self, other, op='/'):
        """Internal function supporting "`/`" and "`/=`"."""

        # Explicitly disallow division by an array; this might work by accident
        if isinstance(other, np.ndarray):
            raise TypeError(f"unsupported operand type(s) for {op}: 'Tabulation' and "
                            "'numpy.ndarray'")

        if isinstance(other, Tabulation):
            domain = self.domain()
            if other._x[0] > domain[0] or other._x[-1] < domain[1]:
                raise ZeroDivisionError('domain of divisor does not span that of '
                                        'dividend')
            other = other.clip(*domain)
            new_x = Tabulation._xmerge(self.x, other.x)
            numer = self(new_x)
            denom = other(new_x)
        else:
            new_x = self._x
            numer = self._y
            denom = other

        if np.any(denom == 0.):
            raise ZeroDivisionError('division by zero')

        return (new_x, numer / denom)

    def __truediv__(self, other):
        """Divide, supporting the "`/`" operator.

        Parameters:
            other (Tabulation, float, or int): The divisor.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            TypeError: If a Tabulation cannot be divided by an object of the given type.
            ZeroDivisionError: If the divisor contains zero or has a narrower domain than
                this Tabulation.

        Notes:
            The returned Tabulation inherits its domain from the numerator. If `other` is
            a Tabulation, its domain must span that of the numerator; otherwise, this is a
            divide-by-zero operation. The ratio is evaluated at the union of the
            Tabulations' x-coordinates within the new domain. Because interpolation is
            linear, values between the new x-coordinates will have reduced accuracy; if
            this is a concern, subsample one of the Tabulations first.
        """

        new_x, new_y = self._divide(other)
        return Tabulation(new_x, new_y)

    def __itruediv__(self, other):
        """In-place divide, supporting the "`/=`" operator.

        Parameters:
            other (float or int): The divisor.

        Returns:
            Tabulation: The current Tabulation, mutated with the new values.

        Raises:
            TypeError: If a Tabulation cannot be divided by an object of the given type.
            ZeroDivisionError: If the divisor contains zero or has a narrower domain than
                this Tabulation.

        Notes:
            The returned Tabulation preserves has an unchanged domain. If `other` is a
            Tabulation, its domain must span that of the given Tabulation; otherwise, this
            is a divide-by-zero operation. The ratio is evaluated at the union of the
            Tabulations' x-coordinates within the domain. Because interpolation is linear,
            values between the new x-coordinates will have reduced accuracy; if this is a
            concern, subsample one of the Tabulations first.
        """

        new_x, new_y = self._divide(other, op='/=')
        return self._update(new_x, new_y)

    def __add__(self, other):
        """Add two Tabulations, supporting the "`+`" operator.

        Parameters:
            other (Tabulation): The Tabulation to add.

        Returns:
            Tabulation: The new Tabulation, sampled at the union of the x-coordinates of
            both Tabulations.
        """

        if not isinstance(other, Tabulation):
            raise TypeError("unsupported operand type(s) for +: 'Tabulation' and "
                            f"'{type(other)}'")

        new_x = Tabulation._xmerge(self._x, other._x)
        return Tabulation(new_x, self(new_x) + other(new_x))

    def __iadd__(self, other):
        """In-place addition, supporting the "`+=`" operator.

        Parameters:
            other (Tabulation): The Tabulation to add.

        Returns:
            Tabulation: The current Tabulation, mutated with the new values. It is sampled
            at the union of the x-coordinates of both Tabulations.
        """

        if not isinstance(other, Tabulation):
            raise TypeError("unsupported operand type(s) for +=: 'Tabulation' and "
                            f"'{type(other)}'")

        new_x = Tabulation._xmerge(self._x, other._x)
        return Tabulation(new_x, self(new_x) + other(new_x))

    def __sub__(self, other):
        """Subtract two Tabulations, supporting the "`-`" operator.

        Parameters:
            other (Tabulation): The Tabulation to subtract.

        Returns:
            Tabulation: The new Tabulation, sampled at the union of the x-coordinates of
            both Tabulations.
       """

        if not isinstance(other, Tabulation):
            raise TypeError("unsupported operand type(s) for -: 'Tabulation' and "
                            f"'{type(other)}'")

        new_x = Tabulation._xmerge(self._x, other._x)
        return Tabulation(new_x, self(new_x) - other(new_x))

    def __isub__(self, other):
        """In-place addition, supporting the "`+=`" operator.

        Parameters:
            other (Tabulation): The Tabulation to subtract.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Returns:
            Tabulation: The current Tabulation, mutated with new values. It is sampled at
            the union of the x-coordinates of both Tabulations.
        """

        if not isinstance(other, Tabulation):
            raise TypeError("unsupported operand type(s) for -=: 'Tabulation' and "
                            f"'{type(other)}'")

        new_x = Tabulation._xmerge(self._x, other._x)
        return Tabulation(new_x, self(new_x) - other(new_x))

    def __eq__(self, other):
        """True if two Tabulations are everywhere equal.

        Parameters:
            other (Tabulation): The Tabulation to compare.

        Returns:
            Tabulation: True if the Tabulations are equal.
        """

        if not isinstance(other, Tabulation):
            return False

        new_x = Tabulation._xmerge(self.x, other.x)
        y1 = self(new_x)
        y2 = other(new_x)
        return np.all(y1 == y2)

    def __getitem__(self, indx):
        """An element or slice of this Tabulation using NumPy index notation.

        This definition allows Python/NumPy indexing notation to be applied to a
        Tabulation using square brackets "`[]`".

        Most of the ways of indexing a NumPy array are supported. If the index is a single
        integer, the value of y at that index is returned. Otherwise, a new Tabulation is
        returned containing only the selected elements. For example, `tab[:10]` is a new
        Tabulation containing the first ten elements of Tabulation `tab`.

        Parameters:
            indx (int, array, list, or slice): Index to apply.

        Returns:
            (float or Tabulation): If the index is a single integer, the value of y at
            that index is returned. Otherwise, a new Tabulation including the selected
            elements of the x and y arrays is returned.

        Raises:
            IndexError: If the index has an invalid value or type.
            ValueError: If the set of elements of the Tabulation selected by the index do
                not represent a valid Tabulation.
        """

        if isinstance(indx, numbers.Integral):
            return self.y[indx]

        return Tabulation(self.x[indx], self.y[indx])

    def __len__(self):
        """Length of this Tabulation.

        This definition supports the use of `len(tab)` to obtain the number of elements in
        Tabulation `tab`.

        Returns:
            int: Number of elements in this Tabulation.
        """

        return self._length

    def __str__(self):
        """Brief string representation of this Tabulation.

        This definition supports the use of `str(tab)` to obtain a brief string describing
        the contents of Tabulation `tab`.

        Returns:
            str: Brief string representation of this Tabulation.
        """

        if self._length <= 4:
            return f'Tabulation({self.x}, {self.y})'

        xlo = str(self.x[:2])[:-1].strip()  # strip trailing "]"
        xhi = str(self.x[-2:])[1:].strip()  # strip leading "["
        ylo = str(self.y[:2])[:-1].strip()
        yhi = str(self.y[-2:])[1:].strip()
        return f'Tabulation({xlo} ... {xhi}, {ylo} ... {yhi})'

    def __repr__(self):
        """Brief string representation of this Tabulation.

        This definition supports the use of `repr(tab)` to obtain a brief string
        describing the contents of Tabulation `tab`.

        Returns:
            str: Brief string representation of this Tabulation.
        """

        return self.__str__()

    ########################################
    # Additional methods
    ########################################

    def domain(self):
        """The range of x-coordinates for which values have been provided.

        Returns:
            tuple: A tuple (xmin, xmax).
        """

        return (float(self._xmin), float(self._xmax))

    def clip(self, xmin=None, xmax=None):
        """A Tabulation where the domain is (xmin, xmax).

        Parameters:
            xmin (float, optional): The minimum value of the new x-coordinates; default is
                to retain the existing lower limit.
            xmax (float, optional): The maximum value of the new x-coordinates; default is
                to retain the existing upper limit.

        Returns:
            Tabulation: The new Tabulation, identical to the current Tabulation except
            that the x domain is now restricted to (`xmin`, `xmax`). If either x
            coordinate is outside the current domain, it is set to that limit of the
            domain.

        Raises:
            ValueError: If the clip domain does not overlap with the Tabulation
                domain.
        """

        if xmin is None:
            xmin = self._xmin
        if xmax is None:
            xmax = self._xmax
        new_x = self._xoverlap(np.array((xmin, xmax)))
        return self.resample(new_x)

    def locate(self, yvalue):
        """The x-coordinates where the Tabulation has the given value of y.

        Note that the exact ends of the domain are not checked.

        Parameters:
            yvalue (float): The value to look for.

        Returns:
            list: A list of x-coordinates where the Tabulation equals `yvalue`.
        """

        signs = np.sign(self.y - yvalue)
        mask = (signs[:-1] * signs[1:]) < 0.

        xlo = self.x[:-1][mask]
        ylo = self.y[:-1][mask]

        xhi = self.x[1:][mask]
        yhi = self.y[1:][mask]

        xarray = xlo + (yvalue - ylo)/(yhi - ylo) * (xhi - xlo)
        xlist = list(xarray) + list(self.x[signs == 0])
        xlist = [float(x) for x in xlist]
        xlist.sort()

        return xlist

    def integral(self, xmin=None, xmax=None):
        """The integral of [y dx].

        Parameters:
            xmin (float, optional): The lower limit of the integral; default is to use the
                lower limit of the Tabulation.
            xmax (float, optional): The upper limit of the integral; default is to use the
                upper limit of the Tabulation.

        Returns:
            float: The integral.
        """

        clipped = self.clip(xmin, xmax)
        ybar_x2 = clipped.y[:-1] + clipped.y[1:]
        dx = np.diff(clipped.x)
        return 0.5 * np.sum(ybar_x2 * dx)

    def resample(self, new_x):
        """A new Tabulation re-sampled at the given x-coordinates.

        Parameters:
            new_x (array-like): The new x-coordinates, which must be monotonic.

        Returns:
            Tabulation: A new Tabulation equivalent to the current Tabulation but sampled
            only at the given x-coordinates.

        Raises:
            ValueError: If the x coordinates are not monotonic.

        Notes:
            If the leading or trailing X coordinate corresponds to a non-zero value, then
            there will be a step at that edge. If the leading or trailing X coordinate
            corresponds to a zero value, then there will be a ramp at that edge. The
            resulting Tabulation is trimmed such that the domain does not include any
            zero-valued coordinates except for those necessary to anchor the leading or
            trailing edge.
        """

        if new_x is None:
            # If new_x is None, return a copy of the current tabulation
            return Tabulation(self.x, self.y)

        new_x = np.asarray(new_x, dtype=np.float64)
        mask = new_x[:-1] < new_x[1:]
        if not np.all(mask):
            mask = new_x[:-1] > new_x[1:]
            if not np.all(mask):
                raise ValueError('x-coordinates are not monotonic')
            new_x = new_x[::-1]

        if len(new_x) == 0 or new_x[-1] < self.x[0] or new_x[0] > self.x[-1]:
            # Resample is entirely outside the current domain, so just return a zero
            # Tabulation.
            return Tabulation([0.], [0.])

        return Tabulation(new_x, self(new_x))

    def subsample(self, new_x=None, *, dx=None, n=None):
        """A new Tabulation re-sampled at a list of x-coords plus existing ones.

        Parameters:
            new_x (array-like, optional): The new x-coordinates.
            dx (float, optional): If provided instead of `new_x`, an array of x-values
                uniformly sampled by `dx` within this Tabulation's domain is used instead.
                If `new_x` is specified, this input is ignored.
            n (int, optional): If provided instead of new_x or dx, this is a number that
                will be used to subdivide the domain, and a new x-value will be inserted
                at each new point.

        Returns:
            Tabulation: A new Tabulation equivalent to the current Tabulation but sampled
            at both the existing x-coordinates and the given x-coordinates.

        Notes:
            If none of new_x, dx, and x are specified, this Tabulation is returned.
        """

        if new_x is not None:
            pass
        elif dx is not None:
            xmin = dx * math.ceil(self._xmin / dx)
            new_x = np.arange(xmin, self._xmax, dx)
        elif n is not None:
            (xmin, xmax) = self.domain()
            dx = (xmax - xmin) / n
            new_x = xmin + dx * np.arange(1, n)
        else:
            return self

        new_x = Tabulation._xmerge(new_x, self._x)
        return Tabulation(new_x, self(new_x))

    def x_mean(self, dx=None):
        """The weighted center x coordinate of the Tabulation.

        Parameters:
            dx (float, optional): The minimum, uniform step size to use when evaluating
                the center position. If omitted, no resampling is performed.

        Returns:
            float: The x coordinate that corresponds to the weighted center of the
            function.
        """

        if dx is None:
            resampled = self
        else:
            (x0, x1) = self.domain()
            new_x = np.arange(x0 + dx, x1, float(dx))
            resampled = self.subsample(new_x)

        integ0 = resampled.integral()

        scaled = Tabulation(resampled.x, resampled.x * resampled.y)
        integ1 = scaled.integral()

        return integ1/integ0

    def bandwidth_rms(self, dx=None):
        """The root-mean-square width of the Tabulation.

        This is the mean value of (y * (x - x_mean)**2)**(1/2).

        Parameters:
            dx (float, optional): The minimum, uniform step size to use when evaluating
                the center position. If omitted, no resampling is performed.

        Returns:
            float: The RMS width of the Tabulation.
        """

        if dx is None:
            resampled = self
        else:
            (x0, x1) = self.domain()
            new_x = np.arange(x0 + dx, x1, float(dx))
            resampled = self.subsample(new_x)

        integ0 = resampled.integral()

        scaled = Tabulation(resampled.x, resampled.x * resampled.y)
        integ1 = scaled.integral()

        scaled = Tabulation(scaled.x, scaled.x * scaled.y)
        integ2 = scaled.integral()

        return np.sqrt(((integ2*integ0 - integ1**2) / integ0**2))

    def pivot_mean(self, precision=0.01):
        """The "pivot" mean value of the tabulation.

        The pivot value is the mean value of y(x) d(log(x)). Note all x must be positive.

        Parameters:
            precision (float, optional): The step size at which to resample the
                Tabulation in log space.

        Returns:
            float: The pivot mean of the Tabulation.
        """

        (x0, x1) = self.domain()

        log_x0 = np.log(x0)
        log_x1 = np.log(x1)
        log_dx = np.log(1. + precision)

        new_x = np.exp(np.arange(log_x0, log_x1 + log_dx, log_dx))

        resampled = self.subsample(new_x)
        integ1 = resampled.integral()

        scaled = Tabulation(resampled.x, resampled.y/resampled.x)
        integ0 = scaled.integral()

        return integ1/integ0

    def fwhm(self, fraction=0.5):
        """The full-width-half-maximum of the Tabulation.

        Parameters:
            fraction (float, optional): The fractional height at which to perform the
                measurement. 0.5 corresponds to "half" maximum for a normal FWHM.

        Returns:
            float: The FWHM for the given fractional height.

        Raises:
            ValueError: If the Tabulation does not cross the fractional height exactly
                twice, or if the fraction is outside the range 0 to 1.
        """

        if not 0 <= fraction <= 1:
            raise ValueError('fraction is outside the range 0-1')

        max = np.max(self.y)
        limits = self.locate(max * fraction)
        if len(limits) != 2:
            raise ValueError('Tabulation does not cross fractional height twice')

        return float(limits[1] - limits[0])

    def square_width(self):
        """The square width of the Tabulation.

        The square width is the width of a rectangular function with y value equal
        to the maximum of the original function and having the same area as the original
        function.

        Returns:
            float: The square width of the Tabulation.
        """

        return float(self.integral() / np.max(self.y))

    def quantile(self, q):
        """The specified quantile point within a Tabulation.

        A quantile point is the x-value that divides the Tabulation into two parts such
        that `fraction` of the integral falls below this value and `1-fraction` falls
        above it.

        Parameters:
            q (float): A fractional value between 0 and 1 inclusive.

        Returns:
            float: The x-value corresponding to the quantile value `q`.

        Raises:
            ValueError: If the fraction is outside the range 0 to 1.
        """

        if not 0 <= q <= 1:
            raise ValueError('q is outside the range 0-1')

        y_dx_x2 = np.empty(self._length)
        y_dx_x2[0] = 0.
        y_dx_x2[1:] = (self.y[:-1] + self.y[1:]) * np.diff(self.x)  # 2x each step's area

        cum_y_dx_x2 = np.cumsum(y_dx_x2)
        integ = q * cum_y_dx_x2[-1]
        signs = np.sign(cum_y_dx_x2 - integ)
        cutoffs = np.where(signs[:-1] * signs[1:] <= 0.)[0]
        i = cutoffs[-1]
            # The solution is within the step from x[i] to x[i+1], inclusive

        # Determine the fractional step of the integral from x[i] to x[i+1]
        frac = (integ - cum_y_dx_x2[i]) / (cum_y_dx_x2[i+1] - cum_y_dx_x2[i])

        # The function is linear within this step, so the quantile requires solving a
        # quadratic. Here t is the fractional step of x between 0 and 1, inclusive.
        #   x(t) = x[i] + t * (x[i+1] - x[i])
        #   y(t) = y[i] + t * (y[i+1] - y[i])
        #   integral[0 to t] = y[i] * t + (y[i+1] - y[i]) / 2 * t**2
        # Solve for:
        #   integral(t) = frac * integral(1)

        a = 0.5 * (self.y[i+1] - self.y[i])
        b = self.y[i]
        c = -frac * (a + b)

        if a == 0.:
            t = -c/b
        else:
            sign_b = 1 if b >= 0 else -1
            neg_b_discr = -b - sign_b * np.sqrt(b*b - 4*a*c)
            t = neg_b_discr / (2*a)
            if not 0 <= t <= 1:
                t = 2*c / neg_b_discr

        return self.x[i] + t * (self.x[i+1] - self.x[i])

##########################################################################################
