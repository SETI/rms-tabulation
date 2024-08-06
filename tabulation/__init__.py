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
import numpy as np
from scipy.interpolate import interp1d

try:
    from math import nextafter  # Only in Python 3.9 and later
except ImportError:  # pragma: no cover
    from numpy import nextafter

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = 'Version unspecified'


class Tabulation(object):
    """A class that represents a function by a sequence of linear interpolations.

    The interpolations are defined between points defined by arrays of x and y
    coordinates. The function is treated as equal to zero outside the range of the x
    coordinates, with a step at the provided leading and trailing x coordinates. Although
    optimized to model filter bandpasses and spectral flux, the class is sufficiently
    general to be used in a wide range of applications.

    In general, zero values (either supplied or computed) at either the leading or
    trailing ends are removed. However, if explicitly supplied, one leading and/or
    trailing zero value is considered significant because it anchors the interpolation of
    a ramp at the beginning or end of the domain. For example::

        >>> t1 = Tabulation([2, 4], [10, 10])  # Leading & trailing step function
        >>> t1.domain()
        (2., 4.)
        >>> t1([0,   1,   1.9, 2,   3,   3.9, 4,   5,   6])
        array([ 0.,  0.,  0., 10., 10., 10., 10.,  0.,  0.])

        >>> t2 = Tabulation([0, 2, 4], [0, 10, 10])  # Ramp on leading edge
        >>> t2.domain()
        (0., 4.)
        >>> t2([0,   1,   1.9,  2,   3,   3.9, 4,   5,   6])
        array([ 0.,  5.,  9.5, 10., 10., 10., 10.,  0.,  0.])

    By default it is assumed that the function never has leading or trailing zeros beyond
    the single zero necessary to anchor the interpolation, and the Tabulation object will
    automatically trim any additional leading and/or trailing regions of the domain that
    have purely zero values.

    When mathematical operations are performed on Tabulations, new x-coordinates are
    added as necessary to keep the behavior of step functions. For example::
    """

    def __init__(self, x, y):
        """
        Constructor for a Tabulation object.

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
            x (array-like): The new 1-D array of x-coordinates.
            y (array-like): The new 1-D array of y-coordinates.

        Returns:
            Tabulation: The current Tabulation object mutated with the new arrays.
        """

        x = np.asarray(x, dtype=np.double)
        y = np.asarray(y, dtype=np.double)

        if len(x.shape) != 1:
            raise ValueError("x array is not 1-dimensional")

        if x.shape != y.shape:
            raise ValueError("x and y arrays do not have the same size")

        if len(x) == 0:
            x = np.array([0.])
            y = np.array([0.])

        mask = x[:-1] < x[1:]
        if np.all(mask):
            self.x = x
            self.y = y
        else:
            mask = x[:-1] > x[1:]
            if not np.all(mask):
                raise ValueError("x-coordinates are not monotonic")
            self.x = x[::-1]
            self.y = y[::-1]

        self._trim()

        self.func = None
        return self

    def _update_y(self, new_y):
        """Update a Tabulation in place with a new y array. Trim the result.

        Parameters:
            new_y (array-like): The new 1-D array of y-coordinates.

        Returns:
            Tabulation: The current Tabulation object mutated with the new array.
        """

        y = np.asarray(new_y, dtype=np.double)

        if y.shape != self.x.shape:
            raise ValueError("x and y arrays do not have the same size")

        self.y = y

        self._trim()

        self.func = None
        return self

    def _trim(self):
        """Update a Tabulation in place by deleting leading/trailing zero-valued regions.

        Notes:
            This will create a copy of the x and y coordinates if trimming is necessary,
            and return the original arrays if trimming is not necessary.
        """

        def _trim1(x, y):
            """Strip away the leading end of an (x,y) array pair."""
            if len(x) <= 2:  # 1 or 2 elements, nothing to trim
                return (x, y)

            # Define a mask at the low end
            mask = np.cumsum(y != 0.) != 0

            # Shift left by one to keep last zero
            mask[:-1] = mask[1:]

            if np.all(mask):
                return (x, y)  # Don't make a copy if it's the same array
            return (x[mask], y[mask])

        # Trim the trailing end
        (new_x, new_y) = _trim1(self.x[::-1], self.y[::-1])

        # Trim the leading end
        (new_x, new_y) = _trim1(new_x[::-1], new_y[::-1])

        self.x = new_x
        self.y = new_y

    @staticmethod
    def _xmerge(x1, x2):
        """Return the union of x-coordinates found in each of the given arrays.

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
            raise ValueError("Domains do not overlap")

        # Merge and sort
        sorted = np.sort(np.hstack((x1, x2)))

        # Locate and remove duplicates
        mask = np.append(sorted[:-1] != sorted[1:], True)
        return sorted[mask]

    @staticmethod
    def _xoverlap(x1, x2):
        """Return the union of x-coords that fall within the intersection of the domains.

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
            # t1 trailing is a step and t2 end to the right, add a tiny ramp
            eps_x = nextafter(t1.x[-1], math.inf)
            x1 = np.concatenate((x1, [eps_x]))
            y1 = np.concatenate((y1, [0.]))
        if t2.y[0] != 0 and t1.x[0] < t2.x[0]:
            # t2 leading is a step and t1 starts to the left, add a tiny ramp
            eps_x = nextafter(t2.x[0], -math.inf)
            x2 = np.concatenate(([eps_x], x2))
            y2 = np.concatenate(([0.], y2))
        if t2.y[-1] != 0 and t1.x[-1] > t2.x[-1]:
            # t2 trailing is a step and t1 end to the right, add a tiny ramp
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
        """Return the interpolated value corresponding to an x-coordinate.

        Parameters:
            x (float or array-like): The x-coordinate(s) at which to evaluate the
                Tabulation.

        Returns:
            float or array-like: The value(s) of the interpolated y-coordinates at the
            given x(s).
        """
        # Fill in the 1-D interpolation if necessary
        if self.func is None:
            self.func = interp1d(self.x, self.y, kind="linear",
                                 bounds_error=False, fill_value=0.)

        if np.shape(x):
            return self.func(x)

        return float(self.func(x))

    def __mul__(self, other):
        """Multiply two Tabulations returning a new Tabulation.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, multiply it with the
                current Tabulation at each interpolation point. If a float is given,
                scale the current Tabulation's y-coordinates uniformly.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap.
            ValueError: If the Tabulation can not be multiplied by the given value.

        Notes:
            The new domain is the intersection of the domains of the current Tabulation
            and the given Tabulation. Because the resulting Tabulation is only computed
            at the existing linear interpolation points, and the resulting Tabulation
            is also linearly interpolated, the values between interpolation points will
            not be accurate (a quadratic interpolation would be required).
        """

        if type(other) is type(self):
            new_x = Tabulation._xoverlap(self.x, other.x)
            return Tabulation(new_x, self(new_x) * other(new_x))

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return Tabulation(self.x, self.y * other)

        raise ValueError("Cannot multiply Tabulation by given value")

    def __truediv__(self, other):
        """Divide two Tabulations returning a new Tabulation.

        Parameters:
            other (float): Scale the current Tabulation's y-coordinates uniformly by
                dividing by the given value.

        Raises:
            ValueError: If the Tabulation can not be multiplied by the given value.

        Returns:
            Tabulation: The new Tabulation.
        """

        if np.shape(other) == ():
            return Tabulation(self.x, self.y / other)

        raise ValueError("Cannot divide Tabulation by given value")

    def __add__(self, other):
        """Add two Tabulations returning a new Tabulation.

        Parameters:
            other (Tabulation): The Tabulation to add to the current Tabulation.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap.
            ValueError: If the Tabulation can not be added to the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if type(other) is type(self):
            t1, t2 = self._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return Tabulation(new_x, t1(new_x) + t2(new_x))

        raise ValueError("Cannot add Tabulation by given value")

    def __sub__(self, other):
        """Subtract two Tabulations returning a new Tabulation.

        Parameters:
            other (Tabulation): The Tabulation to subtract from the current Tabulation.

        Returns:
            Tabulation: The new Tabulation.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap.
            ValueError: If the Tabulation can not be subtracted by the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if type(other) is type(self):
            t1, t2 = self._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return Tabulation(new_x, t1(new_x) - t2(new_x))

        raise ValueError("Cannot subtract Tabulation by given value")

    def __imul__(self, other):
        """Multiply two Tabulations in place.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, multiply it with the
                current Tabulation at each interpolation point. If a float is given,
                scale the y-coordinates uniformly.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap.
            ValueError: If the Tabulation can not be multiplied by the given value.

        Notes:
            The new domain is the intersection of the domains of the current Tabulation
            and the given Tabulation. Because the resulting Tabulation is only computed
            at the existing linear interpolation points, and the resulting Tabulation
            is also linearly interpolated, the values between interpolation points will
            not be accurate (a quadratic interpolation would be required).
        """

        if type(other) is type(self):
            t1, t2 = self._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xoverlap(t1.x, t2.x)
            return self._update(new_x, t1(new_x) * t2(new_x))

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return self._update_y(self.y * other)

        raise ValueError("Cannot multiply Tabulation in-place by given value")

    def __itruediv__(self, other):
        """Divide two Tabulations in place.

        Parameters:
            other (float): Scale the current Tabulation's y-coordinates uniformly by
                dividing by the given value.

        Raises:
            ValueError: If the Tabulation can not be divided by the given value.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.
        """

        if np.shape(other) == ():
            return self._update_y(self.y / other)

        raise ValueError("Cannot divide Tabulation in-place by given value")

    def __iadd__(self, other):
        """Add two Tabulations in place.

        Parameters:
            other (Tabulation): The Tabulation to add to the current Tabulation.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap.
            ValueError: If the Tabulation can not be added to the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if type(other) is type(self):
            t1, t2 = Tabulation._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return self._update(new_x, t1(new_x) + t2(new_x))

        raise ValueError("Cannot add Tabulation in-place by given value")

    def __isub__(self, other):
        """Subtract two Tabulations in place.

        Parameters:
            other (Tabulation): The Tabulation to subtract from the current Tabulation.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Raises:
            ValueError: If the domains of the two Tabulations do not overlap.
            ValueError: If the Tabulation can not be subtracted by the given value.

        Notes:
            The new domain is the union of the domains of the current Tabulation and the
            given Tabulation. The resulting Tabulation will have x-coordinates that are
            the union of the x-coordinates of the current Tabulation and the other
            Tabulation. In addition, additional x-coordinates may be added as necessary to
            ensure the proper behavior in the presence of Tabulations with non-zero
            leading or trailing edges.
        """

        if type(other) is type(self):
            t1, t2 = Tabulation._add_ramps_as_necessary(self, other)
            new_x = Tabulation._xmerge(t1.x, t2.x)
            return self._update(new_x, t1(new_x) - t2(new_x))

        raise ValueError("Cannot subtract Tabulation in-place by given value")

########################################
# Additional methods
########################################

    def domain(self):
        """Return the range of x-coordinates for which values have been provided.

        Returns:
            tuple: A tuple (xmin, xmax).
        """

        return (float(self.x[0]), float(self.x[-1]))

    def clip(self, xmin, xmax):
        """Return a Tabulation where the domain is (xmin, xmax).

        Parameters:
            xmin (float): The minimum value of the new x-coordinates.
            xmax (float): The maximum value of the new x-coordinates.

        Returns:
            Tabulation: The new Tabulation, identical to the current Tabulation except
            that the x domain is now (xmin, xmax). If either x coordinate is beyond
            the range of the current domain, it is set to the current edge of the
            domain.

        Raises:
            ValueError: If the clip domain does not overlap with the Tabulation
            domain.
        """

        new_x = Tabulation._xoverlap(self.x, np.array((xmin, xmax)))
        mask = (new_x >= xmin) & (new_x <= xmax)
        return self.resample(new_x[mask])

    def locate(self, yvalue):
        """Return x-coordinates where the Tabulation has the given value of y.

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

    def integral(self):
        """Return the integral of [y dx].

        Returns:
            float: The integral.
        """

        # Make an array consisting of the midpoints between the x-coordinates
        # Begin with an array holding one extra element
        dx = np.empty(self.x.size + 1)

        dx[1:] = self.x         # Load the array shifted right
        dx[0] = self.x[0]       # Replicate the endpoint

        dx[:-1] += self.x       # Add the array shifted left
        dx[-1] += self.x[-1]

        # dx[] is now actually 2x the value at each midpoint.

        # The weight on each value is the distance between the adjacent midpoints
        dx[:-1] -= dx[1:]   # Subtract the midpoints shifted right (not left)

        # dx[] is now actually -2x the correct value of each weight. The last
        # element is to be ignored.

        # The integral is now the sum of the products y * dx
        return float(-0.5 * np.sum(self.y * dx[:-1]))

    def resample(self, new_x):
        """Return a new Tabulation re-sampled at a given list of x-coordinates.

        Parameters:
            new_x (array-like): The new x-coordinates.

        Returns:
            Tabulation: A new Tabulation equivalent to the current Tabulation but sampled
            only at the given x-coordinates.

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

        new_x = np.asarray(new_x, dtype=np.double)

        mask = new_x[:-1] < new_x[1:]
        if not np.all(mask):
            mask = new_x[:-1] > new_x[1:]
            if not np.all(mask):
                raise ValueError("x-coordinates are not monotonic")
            new_x = new_x[::-1]

        if len(new_x) == 0 or new_x[-1] < self.x[0] or new_x[0] > self.x[-1]:
            # Resample is entirely outside the current domain, so just return a zero
            # Tabulation.
            return Tabulation([0.], [0.])

        return Tabulation(new_x, self(new_x))

    def subsample(self, new_x):
        """Return a new Tabulation re-sampled at a list of x-coords plus existing ones.

        Parameters:
            new_x (array-like): The new x-coordinates.

        Returns:
            Tabulation: A new Tabulation equivalent to the current Tabulation but sampled
            at both the existing x-coordinates and the given x-coordinates.

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

        new_x = Tabulation._xmerge(new_x, self.x)
        return Tabulation(new_x, self(new_x))

    def x_mean(self, dx=None):
        """Return the weighted center x coordinate of the Tabulation.

        Parameters:
            dx (float, optional): The minimum, uniform step size to use when evaluating
                the center position. If omitted, no resampling is performed.

        Returns:
            float: The x coordinate that corresponds to the weighted center of the
                function.
        """

        self._trim()

        if dx is None:
            # y cannot be a shallow copy...
            resampled = Tabulation(self.x, self.y.copy())
        else:
            (x0, x1) = self.domain()
            new_x = np.arange(x0 + dx, x1, dx).astype("float")
            resampled = self.subsample(new_x)

        integ0 = resampled.integral()

        # ...because we change y in-place
        resampled.y *= resampled.x
        integ1 = resampled.integral()

        return integ1/integ0

    def bandwidth_rms(self, dx=None):
        """Return the root-mean-square width of the Tabulation.

        This is the mean value of (y * (x - x_mean)**2)**(1/2).

        Parameters:
            dx (float, optional): The minimum, uniform step size to use when evaluating
                the center position. If omitted, no resampling is performed.

        Returns:
            float: The RMS width of the Tabulation.
        """

        self._trim()

        if dx is None:
            # y cannot be a shallow copy...
            resampled = Tabulation(self.x, self.y.copy())
        else:
            (x0, x1) = self.domain()
            new_x = np.arange(x0 + dx, x1, dx).astype("float")
            resampled = self.subsample(new_x)

        integ0 = resampled.integral()

        # ...because we change y in-place
        resampled.y *= resampled.x
        integ1 = resampled.integral()

        resampled.y *= resampled.x          # ...twice!
        integ2 = resampled.integral()

        return np.sqrt(((integ2*integ0 - integ1**2) / integ0**2))

    def pivot_mean(self, precision=0.01):
        """Return the "pivot" mean value of the tabulation.

        The pivot value is the mean value of y(x) d(log(x)).
        Note all x must be positive.

        Parameters:
            precision (float, optional): The step size at which to resample the
                Tabulation in log space.

        Returns:
            float: The pivot mean of the Tabulation.
        """

        self._trim()
        (x0, x1) = self.domain()

        log_x0 = np.log(x0)
        log_x1 = np.log(x1)
        log_dx = np.log(1. + precision)

        new_x = np.exp(np.arange(log_x0, log_x1 + log_dx, log_dx))

        resampled = self.subsample(new_x)
        integ1 = resampled.integral()

        resampled.y /= resampled.x
        integ0 = resampled.integral()

        return integ1/integ0

    def fwhm(self, fraction=0.5):
        """Return the full-width-half-maximum of the Tabulation.

        Parameters:
            fraction (float, option): The fractional height at which to perform the
                measurement. 0.5 corresponds to "half" maximum for a normal FWHM.

        Returns:
            float: The FWHM for the given fractional height.
        """

        max = np.max(self.y)
        limits = self.locate(max * fraction)
        if len(limits) != 2:
            raise ValueError("Tabulation does not cross fractional height twice")
        return float(limits[1] - limits[0])

    def square_width(self):
        """Return the square width of the Tabulation.

        The square width is the width of a rectangular function with y value equal
        to the maximum of the original function and having the same area as the original
        function.

        Returns:
            float: The square width of the Tabulation.
        """

        return float(self.integral() / np.max(self.y))
