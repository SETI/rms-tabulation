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

    def __init__(self, x, y):
        """Constructor for a Tabulation object.

        Parameters:
            x (array-like): A 1-D array of x-coordinates, which must be monotonic (either
                increasing or decreasing).
            y (array-like): A 1-D array of y-values, given in the same order as the
                x-coordinates.
        """

        self._update(x, y)

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

        y = np.asarray(y, dtype=np.float64)     # makes a copy only if necessary

        xx = np.asarray(x, dtype=np.float64)
        copied = xx is not x
        x = xx

        if len(x.shape) != 1:
            raise ValueError('x array is not 1-dimensional')

        if x.shape != y.shape:
            raise ValueError('x and y arrays do not have the same size')

        if not x.size:
            return self._update([0.], [0.])

        # Swap X-coordinates to increasing
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]

        # Trim...
        nonzeros = np.where(y)[0]
        if nonzeros.size:

            # Slice out the endpoints and their adjacent zeros
            first = nonzeros[0]
            last = nonzeros[-1]
            if first > 0:
                first -= 1
            if last < x.size - 1:
                last += 1

            x = x[first:last+1]
            y = y[first:last+1]

            # Make sure the sequence is monotonic but tolerate duplicates for now
            mask = x[:-1] <= x[1:]
            if not np.all(mask):
                raise ValueError('x-coordinates are not strictly monotonic')

            # Separate duplicated x by epsilon, shifting the one with y closer to zero
            dups = np.where(x[:-1] == x[1:])[0]
            if dups.size and not copied:  # make a copy so user's input array is unchanged
                x = x.copy()

            for i in dups:
                if abs(y[i]) < abs(y[i+1]):
                    x[i] = nextafter(x[i], -math.inf)
                else:
                    x[i+1] = nextafter(x[i], math.inf)

        self.x = x
        self.y = y
        self.x.flags.writeable = False
        self.y.flags.writeable = False
        self.func = None
        return self

    @staticmethod
    def _xmerge(x1, x2):
        """The union of x-coordinates found in each of the given arrays.

        Parameters:
            x1 (array-like): The first array of x-coordinates.
            x2 (array-like): The second array of x-coordinates.

        Returns:
            np.array: The merged array of x-coordinates.

        Raises:
            ValueError: If the domains do not overlap.

        Notes:
            The domains must have some overlap. The resulting domain will range from the
            minimum of the two arrays to the maximum of the two arrays.
        """

        # Confirm overlap
        if x1[0] > x2[-1] or x2[0] > x1[-1]:
            raise ValueError('domains do not overlap')

        # Merge and sort
        sorted = np.sort(np.hstack((x1, x2)))

        # Locate and remove duplicates
        mask = np.append(sorted[:-1] != sorted[1:], True)
        return sorted[mask]

    @staticmethod
    def _xoverlap(x1, x2):
        """The union of x-coords that fall within the intersection of the domains.

        Parameters:
            x1 (array-like): The first array of x-coordinates.
            x2 (array-like): The second array of x-coordinates.

        Returns:
            np.array: The merged array of x-coordinates, limited to those values that
            fall within the intersection of the domains of the two arrays.

        Raises:
            ValueError: If the domains do not overlap.

        Notes:
            The domains must have some overlap. The resulting domain will include only
            the region where the two arrays intersect.
        """

        new_x = Tabulation._xmerge(x1, x2)
        mask = (new_x >= max(x1[0], x2[0])) & (new_x <= min(x1[-1], x2[-1]))
        return new_x[mask]

    @staticmethod
    def _add_ramps_as_necessary(t1, t2):
        """Create new Tabulations as necessary to provide leading/trailing ramps.

        Given two Tabulations, either of which might have a "step" on the leading or
        trailing edge, this function looks at the overlap and adds a microstep if
        necessary to continue to have a step after the Tabulation domains are merged.

        For example, if t1 has x=(5, 7) and y=(1, 1), it has a step at 5 and another step
        at 7. If t2 has x=(4, 5, 6, 7) and y=(0, 1, 1, 0), it has a ramp from 4 to 5 and
        a ramp at 6 to 7. If we try to perform a mathematical operation that combines
        these two Tabulations in some way, t1's step will be changed, incorrectly, to
        a ramp unless a step is forced. We force a step by adding x coordinates at
        the smallest possible increments before or after the step edges, essentially
        creating an infinitesimally-wide ramp. In this case we would create a new
        t1 where x=(5-eps, 5, 7) and y=(0, 1, 1).

        Parameters:
            t1 (Tabulation): The first Tabulation
            t2 (Tabulation): The second Tabulation

        Returns:
            Tabulation, Tabulation: The new Tabulations, if needed, or the original
            Tabulations if not.
        """

        x1 = t1.x
        y1 = t1.y
        x2 = t2.x
        y2 = t2.y

        if t1.y[0] != 0 and t2.x[0] < t1.x[0]:
            # t1 leading is a step and t2 starts to the left, add a tiny ramp
            eps_x = nextafter(t1.x[0], -math.inf)
            x1 = np.concatenate(([eps_x], x1))
            y1 = np.concatenate(([0.], y1))
        if t1.y[-1] != 0 and t2.x[-1] > t1.x[-1]:
            # t1 trailing is a step and t2 ends to the right, add a tiny ramp
            eps_x = nextafter(t1.x[-1], math.inf)
            x1 = np.concatenate((x1, [eps_x]))
            y1 = np.concatenate((y1, [0.]))
        if t2.y[0] != 0 and t1.x[0] < t2.x[0]:
            # t2 leading is a step and t1 starts to the left, add a tiny ramp
            eps_x = nextafter(t2.x[0], -math.inf)
            x2 = np.concatenate(([eps_x], x2))
            y2 = np.concatenate(([0.], y2))
        if t2.y[-1] != 0 and t1.x[-1] > t2.x[-1]:
            # t2 trailing is a step and t1 ends to the right, add a tiny ramp
            eps_x = nextafter(t2.x[-1], math.inf)
            x2 = np.concatenate((x2, [eps_x]))
            y2 = np.concatenate((y2, [0.]))

        if x1 is not t1.x or y1 is not t1.y:
            t1 = Tabulation(x1, y1)
        if x2 is not t2.x or y2 is not t2.y:
            t2 = Tabulation(x2, y2)

        return t1, t2

    ########################################
    # Standard operators
    ########################################

    def __call__(self, x):
        """The interpolated value corresponding to an x-coordinate.

        This definition allows any Tabulation to be treated as a function. So if `tab` is
        a Tabulation, `tab(x)` returns the value of that Tabulation evaluated at `x`.

        Parameters:
            x (float or array-like): The x-coordinate(s) at which to evaluate the
                Tabulation.

        Returns:
            float or array-like: The value(s) of the interpolated y-coordinates at the
            given x(s).
        """

        # Fill in the 1-D interpolation if necessary
        if self.func is None:
            self.func = interp1d(self.x, self.y, kind='linear',
                                 bounds_error=False, fill_value=0.)

        value = self.func(x)
        if np.shape(x):
            return value

        return float(value[()])

    def __mul__(self, other):
        """Multiply two Tabulations returning a new Tabulation.

        This definition allows a Tabulation to be multiplied by a constant or two
        Tabulations to be multiplied together using the "`*`" operator, yielding a new
        Tabulation.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, multiply it with the
                current Tabulation at each interpolation point. If a float is given,
                scale the current Tabulation's y-coordinates uniformly.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap, or if the
                Tabulation can not be multiplied by the given value.

        Notes:
            The new domain is the intersection of the domains of the current Tabulation
            and the given Tabulation. Because the resulting Tabulation is only computed
            at the existing linear interpolation points, and the resulting Tabulation
            is also linearly interpolated, the values between interpolation points will
            not be accurate (a quadratic interpolation would be required).
        """

        if isinstance(other, Tabulation):
            new_x = Tabulation._xoverlap(self.x, other.x)
            return Tabulation(new_x, self(new_x) * other(new_x))

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return Tabulation(self.x, self.y * other)

        raise ValueError('cannot multiply Tabulation by given value')

    def __truediv__(self, other):
        """Divide two Tabulations returning a new Tabulation.

        This definition allows a Tabulation to be divided by a constant using the "`/`"
        operator, yielding a new Tabulation.

        Parameters:
            other (float): Scale the current Tabulation's y-coordinates uniformly by
                dividing by the given value.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            ValueError: If the Tabulation can not be multiplied by the given value.
        """

        if np.shape(other) == ():
            return Tabulation(self.x, self.y / other)

        raise ValueError('cannot divide Tabulation by given value')

    def __add__(self, other):
        """Add two Tabulations returning a new Tabulation.

        This definition allows two Tabulations to be added together using a "`+`"
        operator, yielding a new Tabulation.

        Parameters:
            other (Tabulation): The Tabulation to add to the current Tabulation.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap, or if the
                Tabulation can not be added to the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if isinstance(other, Tabulation):
            t1, t2 = self._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return Tabulation(new_x, t1(new_x) + t2(new_x))

        raise ValueError('cannot add Tabulation by given value')

    def __sub__(self, other):
        """Subtract two Tabulations returning a new Tabulation.

        This definition allows two Tabulations to be subtracted using a "`-`" operator,
        yielding a new Tabulation.

        Parameters:
            other (Tabulation): The Tabulation to subtract from the current Tabulation.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap, or if the
                Tabulation can not be subtracted by the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if isinstance(other, Tabulation):
            t1, t2 = self._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return Tabulation(new_x, t1(new_x) - t2(new_x))

        raise ValueError('cannot subtract Tabulation by given value')

    def __imul__(self, other):
        """Multiply two Tabulations in place.

        This definition allows the in-place multiplication of a Tabulation by a constant
        or another Tabulation using the "`*=`" operator.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, multiply it with the
                current Tabulation at each interpolation point. If a float is given,
                scale the y-coordinates uniformly.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap, or if the
                Tabulation can not be multiplied by the given value.

        Notes:
            The new domain is the intersection of the domains of the current Tabulation
            and the given Tabulation. Because the resulting Tabulation is only computed
            at the existing linear interpolation points, and the resulting Tabulation
            is also linearly interpolated, the values between interpolation points will
            not be accurate. Note that you can use subsample() to improve precision.
        """

        if isinstance(other, Tabulation):
            t1, t2 = self._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xoverlap(t1.x, t2.x)
            return self._update(new_x, t1(new_x) * t2(new_x))

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return self._update(self.x, self.y * other)

        raise ValueError('cannot multiply Tabulation in-place by given value')

    def __itruediv__(self, other):
        """Divide two Tabulations in place.

        This definition allows the in-place division of a Tabulation by a constant,
        using the "`/=`" operator.

        Parameters:
            other (float): Scale the current Tabulation's y-coordinates uniformly by
                dividing by the given value.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Raises:
            ValueError: If the Tabulation can not be divided by the given value.
        """

        if np.shape(other) == ():
            return self._update(self.x, self.y / other)

        raise ValueError('cannot divide Tabulation in-place by given value')

    def __iadd__(self, other):
        """Add two Tabulations in place.

        This definition allows the in-place addition of one Tabulation to another, using
        the "`+=`" operator.

        Parameters:
            other (Tabulation): The Tabulation to add to the current Tabulation.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap, or if the
                Tabulation can not be added to the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if isinstance(other, Tabulation):
            t1, t2 = Tabulation._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return self._update(new_x, t1(new_x) + t2(new_x))

        raise ValueError('cannot add Tabulation in-place by given value')

    def __isub__(self, other):
        """Subtract two Tabulations in place.

        This definition allows the in-place subtraction of one Tabulation from another,
        using the "`-=`" operator.

        Parameters:
            other (Tabulation): The Tabulation to subtract from the current Tabulation.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap, or if the
                Tabulation can not be subtracted by the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if isinstance(other, Tabulation):
            t1, t2 = Tabulation._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return self._update(new_x, t1(new_x) - t2(new_x))

        raise ValueError('cannot subtract Tabulation in-place by given value')

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

        return len(self.x)

    def __str__(self):
        """Brief string representation of this Tabulation.

        This definition supports the use of `str(tab)` to obtain a brief string describing
        the contents of Tabulation `tab`.

        Returns:
            str: Brief string representation of this Tabulation.
        """

        if len(self.x) <= 4:
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

        if self.x.size == 1:
            x = float(self.x[0])
            return (x, x)

        # Strip bounding x-values that are within 3 * epsilon of the adjacent x
        xmin = self.x[0]
        if self.y[0] == 0. and self.y[1] != 0.:
            limit = nextafter(self.x[1], -math.inf, steps=3)
            if self.x[0] > limit:
                xmin = self.x[1]

        xmax = self.x[-1]
        if self.y[-1] == 0. and self.y[-2] != 0.:
            limit = nextafter(self.x[-2], math.inf, steps=3)
            if self.x[-1] < limit:
                xmax = self.x[-2]

        return (float(xmin), float(xmax))

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
            xmin = self.x[0]
        if xmax is None:
            xmax = self.x[-1]
        new_x = Tabulation._xoverlap(self.x, np.array((xmin, xmax)))
        mask = (new_x >= xmin) & (new_x <= xmax)
        return self.resample(new_x[mask])

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
        """A new Tabulation re-sampled at a given list of x-coordinates.

        Parameters:
            new_x (array-like): The new x-coordinates.

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
            return Tabulation(self.x, self.y.copy())

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
            xmin = dx * math.ceil(self.x[0] / dx)
            new_x = np.arange(xmin, self.x[-1], dx)
        elif n is not None:
            (xmin, xmax) = self.domain()
            dx = (xmax - xmin) / n
            new_x = xmin + dx * np.arange(1, n)
        else:
            return self

        new_x = Tabulation._xmerge(new_x, self.x)
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

        y_dx_x2 = np.empty(len(self))
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
