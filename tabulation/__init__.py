##########################################################################################
# tabulation/__init__.py
##########################################################################################
"""
Tabulation class,
PDS Ring-Moon Systems Node, SETI Institute

The Tabulation class represents a mathematical function by a sequence of linear
interpolations between points defined by arrays of x and y coordinates. See the
documentation for the Tabulation class for full details.
"""

import numpy as np
from scipy.interpolate import interp1d

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = 'Version unspecified'


class Tabulation(object):
    """A class that represents a function by a sequence of linear interpolations.

    The interpolations are defined between points defined by arrays of x and y
    coordinates. The function is treated as equal to zero outside the range of the x
    coordinates, with a step at the provided leading and trailing x coordinates.

    However, if explicitly supplied, one leading and/or trailing zero value is considered
    significant because it anchors the interpolation at the beginning or end of the
    domain. You should not provide more than one leading and/or trailing zero. For
    example::

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

    By default it is assumed that the function never has leading or trailing zeros beyond
    the single zero necessary to anchor the interpolation, and the Tabulation object will
    automatically trim any additional leading and/or trailing regions of the domain that
    have purely zero values after many operations. This is done to improve space and time
    efficiency.

    However, while reasonable for most applications, there are times where it is important
    to have the domain include some amount of zero values, as trimming affects
    computations such as `mean` and `pivot_mean`. In these cases, the best workaround is
    to supply extremely small but non-zero values for those regions.

    Note that you can not generally mix step- and ramp-style Tabulations in mathematical
    operations such as addition or multiplication.
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
        """Update a Tabulation in place with new x and y arrays.

        Parameters:
            x (array-like): The new 1-D array of x-coordinates.
            y (array-like): The new 1-D array of y-coordinates.

        Returns:
            Tabulation: The current Tabulation object mutated with the new arrays.
        """

        x = np.asarray(x, dtype=np.double)
        y = np.asarray(y, dtype=np.double)
        sorted = np.sort(x)

        if len(x.shape) != 1:
            raise ValueError("x array is not 1-dimensional")

        if x.shape != y.shape:
            raise ValueError("x and y arrays do not have the same size")

        if np.all(sorted == x):
            self.x = x
            self.y = y
        elif np.all(sorted == x[::-1]):
            self.x = x[::-1]
            self.y = y[::-1]
        else:
            raise ValueError("x-coordinates are not monotonic")

        self.func = None
        return self

    def _update_y(self, new_y):
        """Update a Tabulation in place with a new y array.

        Parameters:
            new_y (array-like): The new 1-D array of y-coordinates.

        Returns:
            Tabulation: The current Tabulation object mutated with the new array.
        """

        y = np.asarray(new_y, dtype=np.double)

        if y.shape != self.x.shape:
            raise ValueError("x and y arrays do not have the same size")

        self.y = y
        self.func = None
        return self

    def _trim(self):
        """Update a Tabulation in place by deleting leading/trailing zero-valued regions.

        Returns:
            Tabulation: The current Tabulation object mutated with the unused regions
            removed.

        Notes:
            This will always create a copy of the x and y coordinates.
        """

        def _trim1(x, y):
            """Strip away the leading end of an (x,y) array pair."""
            # Define a mask at the low end
            mask = np.cumsum(y != 0.) != 0

            # Shift left by one to keep last zero
            mask[:-1] = mask[1:]

            return (x[mask], y[mask])

        # Trim the trailing end
        (new_x, new_y) = _trim1(self.x[::-1], self.y[::-1])

        # Trim the leading end
        (new_x, new_y) = _trim1(new_x[::-1], new_y[::-1])

        return self._update(new_x, new_y)

    @staticmethod
    def _xmerge(x1, x2):
        """Return the union of x-coordinates found in each of the given arrays.

        Parameters:
            x1 (array-like): The first array of x-coordinates.
            x2 (array-like): The second array of x-coordinates.

        Returns:
            np.array: The merged array of x-coordinates.

        Notes:
            The domains must have some overlap. The resulting domain will range from the
            minimum of the two arrays to the maximum of the two arrays.
        """

        # Confirm overlap
        if x1[0] > x2[-1] or x2[0] > x1[-1]:
            raise ValueError("domains do not overlap")

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

        Notes:
            The domains must have some overlap. The resulting domain will include only
            the region where the two arrays intersect.
        """

        new_x = Tabulation._xmerge(x1, x2)
        mask = (new_x >= max(x1[0], x2[0])) & (new_x <= min(x1[-1], x2[-1]))
        return new_x[mask]

    def _check_step_ramp_compatibility(self, other):
        """Raise an exception if this and other have different step- and ramp-styles.

        Parameters:
            other (Tabulation): The second Tabulation object to compare against.

        Raises:
            ValueError: If the leading and/or trailing style of this Tabulation and other
            do not match.
        """

        if (self.y[0] == 0) != (other.y[0] == 0):
            raise ValueError("Incompatible leading step/ramp styles")
        if (self.y[-1] == 0) != (other.y[-1] == 0):
            raise ValueError("Incompatible trailing step/ramp styles")

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

        Notes:
            The new domain is the intersection of the domains of the current Tabulation
            and the given Tabulation.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xoverlap(self.x, other.x)
            return Tabulation(new_x, self(new_x) * other(new_x))

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return Tabulation(self.x, self.y * other)

        raise ValueError("Cannot multiply Tabulation by given value")

    def __truediv__(self, other):
        """Divide two Tabulations returning a new Tabulation.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, divide the current
                Tabulation by this Tabulation at each interpolation point. If a float is
                given, scale the current Tabulation's y-coordinates uniformly.

        Returns:
            Tabulation: The new Tabulation.

        Notes:
            The new domain is the intersection of the domains of the current Tabulation
            and the given Tabulation.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xoverlap(self.x, other.x)
            return Tabulation(new_x, self(new_x) / other(new_x))

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return Tabulation(self.x, self.y / other)

        raise ValueError("Cannot divide Tabulation by given value")

    def __add__(self, other):
        """Add two Tabulations returning a new Tabulation.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, add it to the current
                Tabulation at each interpolation point. If a float is given, add it to
                each of the current Tabulation's y-coordinates uniformly.

        Returns:
            Tabulation: The new Tabulation.

        Notes:
            The new domain is the union of the domains of the current Tabulation and
            the given Tabulation.

            A constant added to a Tabulation will still return zero outside the domain.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xmerge(self.x, other.x)
            return Tabulation(new_x, self(new_x) + other(new_x))

        # Otherwise just shift the y-values
        elif np.shape(other) == ():
            return Tabulation(self.x, self.y + other)

        raise ValueError("Cannot add Tabulation by given value")

    def __sub__(self, other):
        """Subtract two Tabulations returning a new Tabulation.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, subtract it from the
                current Tabulation at each interpolation point. If a float is given,
                subtract it from each of the current Tabulation's y-coordinates uniformly.

        Returns:
            Tabulation: The new Tabulation.

        Notes:
            The new domain is the union of the domains of the current Tabulation and
            the given Tabulation.

            A constant subtracted from a Tabulation will still return zero outside the
            domain.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xmerge(self.x, other.x)
            return Tabulation(new_x, self(new_x) - other(new_x))

        # Otherwise just shift the y-values
        elif np.shape(other) == ():
            return Tabulation(self.x, self.y - other)

        raise ValueError("Cannot subtract Tabulation by given value")

    def __imul__(self, other):
        """Multiply two Tabulations in place.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, multiply it with the
                current Tabulation at each interpolation point. If a float is given,
                scale the y-coordinates uniformly.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Notes:
            The new domain is the intersection of the given domains.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xoverlap(self.x, other.x)
            return self._update(new_x, self(new_x) * other(new_x))._trim()

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return self._update_y(self.y * other)

        raise ValueError("Cannot multiply Tabulation in-place by given value")

    def __itruediv__(self, other):
        """Divide two Tabulations in place.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, divide the current
                Tabulation by this Tabulation at each interpolation point. If a float is
                given, scale the y-coordinates uniformly.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Notes:
            The new domain is the intersection of the given domains.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xoverlap(self.x, other.x)
            return self._update(new_x, self(new_x) / other(new_x))._trim()

        # Otherwise just scale the y-values
        elif np.shape(other) == ():
            return self._update_y(self.y / other)

        raise ValueError("Cannot divide Tabulation in-place by given value")

    def __iadd__(self, other):
        """Add two Tabulations in place.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, add it to the current
                Tabulation at each interpolation point. If a float is given, add it to
                each of the y-coordinates uniformly.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Notes:
            The new domain is the union of the given domains.

            A constant added to a Tabulation will still return zero outside the domain.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xmerge(self.x, other.x)
            return self._update(new_x, self(new_x) + other(new_x))

        # Otherwise just shift the y-values
        elif np.shape(other) == ():
            return self._update_y(self.y + other)

        raise ValueError("Cannot add Tabulation in-place by given value")

    def __isub__(self, other):
        """Subtract two Tabulations in place.

        Parameters:
            other (Tabulation or float): If a Tabulation is given, subtract it from the
                current Tabulation at each interpolation point. If a float is given,
                subtract it from each of the y-coordinates uniformly.

        Returns:
            Tabulation: The current Tabulation mutated with the new values.

        Notes:
            The new domain is the union of the given domains.

            A constant subtracted from a Tabulation will still return zero outside the
            domain.
        """

        if type(other) is type(self):
            self._check_step_ramp_compatibility(other)
            new_x = Tabulation._xmerge(self.x, other.x)
            return self._update(new_x, self(new_x) - other(new_x))

        # Otherwise just shift the y-values
        elif np.shape(other) == ():
            return self._update_y(self.y - other)

        raise ValueError("Cannot subtract Tabulation in-place by given value")

########################################
# Additional methods
########################################

    def trim(self):
        """Return a new Tabulation where zero-valued leading/trailing regions are removed.

        Returns:
            Tabulation: A copy of the current Tabulation object with any zero-valued
            leading or trailing regions removed. A single leading or trailing zero
            will be left to anchor the interpolation as necessary.

        Notes:
            Calling this function is not generally necessary, as it is performed
            automatically by various methods.
        """

        # Save the original arrays
        x = self.x
        y = self.y

        # Create a trimmed version
        self._trim()        # operates in-place
        result = Tabulation(self.x, self.y)

        # Restore the original
        self.x = x
        self.y = y

        return result

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
            the range of the current domain, zeros are assumed for the y values if
            the current Tabulation is ramp-style. Otherwise, an exception is raised.

        Raises:
            ValueError: If either x coordinate is beyond the range of the current
            domain, and current Tabulation is not ramp-style.

        Notes:
            If the clip results in a leading or trailing zero value when there was
            not one previously, the Tabulation will now be treated as ramp-style.
        """

        new_x = Tabulation._xmerge(self.x, np.array((xmin, xmax)))
        if new_x[0] < self.x[0] and self.y[0] != 0:
            raise ValueError("Clipping operation changed leading edge to ramp-style")
        if new_x[-1] > self.x[-1] and self.y[-1] != 0:
            raise ValueError("Clipping operation changed trailing edge to ramp-style")
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
            Tabulation: A new Tabulation equivalent to the current Tabulation but
            sampled only at the given x-coordinates.
        """

        if new_x is None:
            # If new_x is None, return a copy of the current tabulation
            return Tabulation(self.x, self.y.copy())

        return Tabulation(new_x, self(new_x))

    def subsample(self, new_x):
        """Return a new Tabulation re-sampled at a list of x-coords plus existing ones.

        Parameters:
            new_x (array-like): The new x-coordinates.

        Returns:
            Tabulation: A new Tabulation equivalent to the current Tabulation but
            sampled only at the given x-coordinates.
        """

        new_x = Tabulation._xmerge(new_x, self.x)
        return Tabulation(new_x, self(new_x))

    def mean(self):
        """Return the mean value of the Tabulation.

        Returns:
            float: The mean across the domain, ignoring leading and trailing zero regions.
        """

        self._trim()
        (x0, x1) = self.domain()

        integ = self.integral()

        return float(integ / (x1-x0))

    # def bandwidth_rms(self):
    #     """Return the root-mean-square width of the Tabulation.

    #     This is the mean value of (y * (x - x_mean)**2)**(1/2).

    #     Returns:
    #         float: The RMS width of the Tabulation.
    #     """

    #     self._trim()
    #     (x0, x1) = self.domain()
    #     x_mean = (x0 + x1) / 2

    #     integ0 = resampled.integral()

    #     resampled.y *= resampled.x          # ...because we change y in-place
    #     integ1 = resampled.integral()

    #     resampled.y *= resampled.x          # ...twice!
    #     integ2 = resampled.integral()

    #     return float(np.sqrt(((integ2*integ0 - integ1**2) / integ0**2)))

    # def pivot_mean(self, precision=0.01):
    #     """Return the "pivot" mean value of the tabulation.

    #     The pivot value is the mean value of y(x) d(log(x)).
    #     Note all x must be positive.

    #     Parameters:
    #         precision (float, optional): The step size at which to resample the
    #             Tabulation.

    #     Returns:
    #         float: The pivot mean of the Tabulation.
    #     """

    #     self._trim()
    #     (x0, x1) = self.domain()

    #     log_x0 = np.log(x0)
    #     log_x1 = np.log(x1)
    #     log_dx = np.log(1. + precision)

    #     new_x = np.exp(np.arange(log_x0, log_x1 + log_dx, log_dx))

    #     resampled = self.subsample(new_x)
    #     integ = resampled.integral()

    #     return float(integ / (log_x1 - log_x0))

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
