########################################
# UNIT TESTS
########################################

from tabulation import Tabulation

import math
import numpy as np
import unittest


class Test_Tabulation(unittest.TestCase):

    def runTest(self):

        tab = Tabulation([], [])
        self.assertEqual(tab.domain(), (0., 0.))
        self.assertEqual(tab(0), 0)
        self.assertEqual(tab(-1), 0)
        self.assertEqual(tab(1), 0)

        # Ramp-style edges
        # xr = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # yr = [0, 0, 0, 1, 2, 6, 1, 0, 0]
        xr = [-1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
        yr = [0.,  0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,  0.,  0.]

        # Step-style edges
        xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
        ys = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

        tabr = Tabulation(xr, yr)
        self.assertEqual(tabr.domain(), (0., 11.))  # Trimmed

        tabs = Tabulation(xs, ys)
        self.assertEqual(tabs.domain(), (1., 10.))  # Not trimmed

        self.assertEqual(4., tabs(4))
        self.assertEqual(4.5, tabs(4.5))
        self.assertEqual(0., tabs(10.000000001))

        reverseds = Tabulation(xs[::-1], ys)  # Reverse X only, should sort ascending
        self.assertEqual(5., reverseds(6))
        self.assertEqual(5.5, reverseds(5.5))
        self.assertEqual(0., reverseds(10.000000001))
        self.assertEqual(reverseds.domain(), (1., 10.))
        reversedr = Tabulation(xr[::-1], yr)  # Reverse X only, should sort ascending
        self.assertEqual(5., reversedr(6))
        self.assertEqual(5.5, reversedr(5.5))
        self.assertEqual(0., reversedr(11.000000001))
        self.assertEqual(reversedr.domain(), (0., 11.))

        self.assertTrue(np.all(np.array([3.5, 4.5, 5.5]) == tabs([3.5, 4.5, 5.5])))
        self.assertTrue(tabs.integral(), 50.)

        # RESAMPLE

        resampled = tabs.resample(None)

        self.assertTrue(np.all(resampled.x == xs))
        self.assertTrue(np.all(resampled.y == ys))

        with self.assertRaises(ValueError) as context:
            tabs.resample([-4, -5, -3])
        self.assertEqual(str(context.exception), "x-coordinates are not monotonic")

        # All off left side
        resampled = tabs.resample([-5, -4])
        self.assertEqual(resampled.domain(), (0., 0.))
        resampled = tabr.resample([-5, -4])
        self.assertEqual(resampled.domain(), (0., 0.))

        # All off right side
        resampled = tabs.resample([50])
        self.assertEqual(resampled.domain(), (0., 0.))
        resampled = tabr.resample([50])
        self.assertEqual(resampled.domain(), (0., 0.))

        # Within existing domain
        resampled = tabs.resample(np.arange(1, 10.1, 0.5))
        self.assertTrue(np.all(resampled.y == resampled.x))
        self.assertEqual(resampled.x.shape, (19,))
        self.assertEqual(resampled.domain(), (1, 10))
        resampled = tabr.resample(np.arange(1, 10.1, 0.5))
        self.assertTrue(np.all(resampled.y == resampled.x))
        self.assertEqual(resampled.x.shape, (19,))
        self.assertEqual(resampled.domain(), (1, 10))
        resampled = tabr.resample(np.arange(10, 0.5, -0.5))
        self.assertTrue(np.all(resampled.y == resampled.x))
        self.assertEqual(resampled.x.shape, (19,))
        self.assertEqual(resampled.domain(), (1, 10))

        # Within existing domain
        resampled = tabs.resample(np.array([1., 10.]))
        self.assertTrue(np.all(resampled.y == resampled.x))
        self.assertEqual(resampled.x.shape, (2,))
        self.assertEqual(resampled.domain(), (1, 10))
        resampled = tabr.resample(np.array([1., 10.]))  # Non-zero wings
        self.assertTrue(np.all(resampled.y == resampled.x))
        self.assertEqual(resampled.x.shape, (2,))
        self.assertEqual(resampled.domain(), (1, 10))
        resampled = tabr.resample(np.array([0., 11.]))  # Zero wings
        self.assertTrue(np.all(resampled.y == np.array([0., 0.])))
        self.assertEqual(resampled.x.shape, (2,))
        self.assertEqual(resampled.domain(), (0, 11))

        # Off left side only, trim
        resampled = tabs.resample(np.array([0., 0.5, 10.]))
        self.assertTrue(np.all(resampled.x == np.array([0.5, 10.])))
        self.assertTrue(np.all(resampled.y == np.array([0., 10.])))
        self.assertEqual(resampled.x.shape, (2,))
        self.assertEqual(resampled.domain(), (0.5, 10))
        resampled = tabr.resample(np.array([-1., 0., 0.5, 10.]))
        self.assertTrue(np.all(resampled.x == np.array([0., 0.5, 10.])))
        self.assertTrue(np.all(resampled.y == np.array([0., 0.5, 10.])))
        self.assertEqual(resampled.x.shape, (3,))
        self.assertEqual(resampled.domain(), (0., 10))

        # Off right side only, trim
        resampled = tabs.resample(np.array([10., 15., 16.]))
        self.assertTrue(np.all(resampled.x == np.array([10., 15.])))
        self.assertTrue(np.all(resampled.y == np.array([10., 0.])))
        self.assertEqual(resampled.x.shape, (2,))
        self.assertEqual(resampled.domain(), (10, 15))
        resampled = tabr.resample(np.array([10., 15., 16.]))
        self.assertTrue(np.all(resampled.x == np.array([10., 15.])))
        self.assertTrue(np.all(resampled.y == np.array([10., 0.])))
        self.assertEqual(resampled.x.shape, (2,))
        self.assertEqual(resampled.domain(), (10, 15))

        # SUBSAMPLE

        subsampled = tabs.subsample(None)

        self.assertTrue(np.all(subsampled.x == xs))
        self.assertTrue(np.all(subsampled.y == ys))

        subsampled = tabs.subsample(np.array([5.2, 5.5, 6., 7.]))
        self.assertTrue(np.all(subsampled.x ==
                               np.array([1., 2., 3., 4., 5., 5.2, 5.5,
                                         6., 7., 8., 9., 10.])))
        self.assertTrue(np.all(subsampled.y == subsampled.x))
        subsampled = tabr.subsample(np.array([5.2, 5.5, 6., 7.]))
        self.assertTrue(np.all(subsampled.x ==
                               np.array([0., 1., 2., 3., 4., 5., 5.2, 5.5,
                                         6., 7., 8., 9., 10., 11.])))
        self.assertTrue(np.all(subsampled.y == np.array([0., 1., 2., 3., 4., 5., 5.2, 5.5,
                                                         6., 7., 8., 9., 10., 0.])))

        subsampled = tabs.subsample(np.array([0., 11.]))
        self.assertTrue(np.all(subsampled.x ==
                               np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.,
                                         9., 10., 11.])))
        self.assertTrue(np.all(subsampled.y == np.array([0., 1., 2., 3., 4., 5., 6., 7.,
                                                         8., 9., 10., 0.])))
        subsampled = tabr.subsample(np.array([0., 11.]))
        self.assertTrue(np.all(subsampled.x ==
                               np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.,
                                         9., 10., 11.])))
        self.assertTrue(np.all(subsampled.y == np.array([0., 1., 2., 3., 4., 5., 6., 7.,
                                                         8., 9., 10., 0.])))

        tab = Tabulation(np.arange(10.) + 0.25, [0, 0, 1, 1, 0, 1, 0, 1, 1, 2])
        subsampled = tab.subsample(dx=1)
        for i in range(2, 10):
            self.assertIn(i, subsampled.x)

        subsampled = tab.subsample(n=16)
        for i in range(2, 9):
            self.assertIn(i+0.75, subsampled.x)

        subsampled = tab.subsample()
        self.assertIs(subsampled, tab)

        # ADDITION

        x1 = [2, 3, 4, 5, 6, 7, 8, 9]  # Large step
        y1 = [1, 2, 3, 4, 5, 4, 3, 2]
        tab1 = Tabulation(x1, y1)
        x2 = [4.5, 5, 6, 7]  # Centered step
        y2 = [1, 2, 3, 1]
        tab2 = Tabulation(x2, y2)
        x3 = [1, 2, 3, 4, 4.3, 6.3, 7.3]  # Large ramp offset to left
        y3 = [0, 1, 2, 3, 4, 5, 0]
        tab3 = Tabulation(x3, y3)
        x4 = [5.3, 6.1, 7.1, 8.1, 8.4]  # Small ramp offset to right
        y4 = [0, 1, 2, 3, 0]
        tab4 = Tabulation(x4, y4)

        off_xlist = np.arange(-1.02, 12.05, .05)

        # This tests the addition of ramps explicitly
        res = tab1 + tab2
        self.assertTrue(np.all(res.x == np.array([2., 3., 4., 4.499999999999999,
                                                  4.5, 5., 6., 7., 7.000000000000001,
                                                  8.,  9])))
        self.assertTrue(np.all(res.y == np.array([1, 2, 3, 3.499999999999999, 4.5,
                                                  6, 8, 5, 3.999999999999999, 3, 2])))
        self.assertTrue(np.all(
            np.abs((tab1(off_xlist) + tab2(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab2 + tab1
        self.assertTrue(
            np.all(np.abs((tab2(off_xlist) + tab1(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab3 + tab4
        self.assertTrue(
            np.all(res.x == np.array([1., 2., 3., 4., 4.3, 5.3, 6.1,
                                      6.3, 7.1, 7.3, 8.1, 8.4])))
        self.assertTrue(
            np.all(res.y-np.array([0., 1., 2., 3., 4., 4.5, 5.9,
                                   6.2, 3., 2.2, 3., 0.])) < 1e-10)
        self.assertTrue(
            np.all(np.abs((tab3(off_xlist) + tab4(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab4 + tab3
        self.assertTrue(
            np.all(np.abs((tab4(off_xlist) + tab3(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab1 + tab3
        self.assertTrue(
            np.all(np.abs((tab1(off_xlist) + tab3(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab3 + tab1
        self.assertTrue(
            np.all(np.abs((tab3(off_xlist) + tab1(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab2 + tab4
        self.assertTrue(
            np.all(np.abs((tab2(off_xlist) + tab4(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab4 + tab2
        self.assertTrue(
            np.all(np.abs((tab4(off_xlist) + tab2(off_xlist) - res(off_xlist))) < 1e-10))

        tab5 = Tabulation(x1, y1)
        tab5 += tab2
        self.assertTrue(
            np.all(np.abs((tab1(off_xlist) + tab2(off_xlist)-tab5(off_xlist))) < 1e-10))

        with self.assertRaises(ValueError) as context:
            _ = tab1 + np.array([])
        self.assertEqual(str(context.exception), "cannot add Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            _ = tab1 + 5
        self.assertEqual(str(context.exception), "cannot add Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 += np.array([])
        self.assertEqual(str(context.exception),
                         "cannot add Tabulation in-place by given value")

        with self.assertRaises(ValueError) as context:
            tab1 += 5
        self.assertEqual(str(context.exception),
                         "cannot add Tabulation in-place by given value")

        # SUBTRACTION

        res = tab1 - tab2
        self.assertTrue(
            np.all(np.abs((tab1(off_xlist)-tab2(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab2 - tab1
        self.assertTrue(
            np.all(np.abs((tab2(off_xlist)-tab1(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab3 - tab4
        self.assertTrue(
            np.all(np.abs((tab3(off_xlist)-tab4(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab4 - tab3
        self.assertTrue(
            np.all(np.abs((tab4(off_xlist)-tab3(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab1 - tab3
        self.assertTrue(
            np.all(np.abs((tab1(off_xlist)-tab3(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab3 - tab1
        self.assertTrue(
            np.all(np.abs((tab3(off_xlist)-tab1(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab2 - tab4
        self.assertTrue(
            np.all(np.abs((tab2(off_xlist)-tab4(off_xlist) - res(off_xlist))) < 1e-10))

        res = tab4 - tab2
        self.assertTrue(
            np.all(np.abs((tab4(off_xlist)-tab2(off_xlist) - res(off_xlist))) < 1e-10))

        tab5 = Tabulation(x1, y1)
        tab5 -= tab2
        self.assertTrue(
            np.all(np.abs((tab1(off_xlist)-tab2(off_xlist)-tab5(off_xlist))) < 1e-10))

        with self.assertRaises(ValueError) as context:
            _ = tab1 - np.array([])
        self.assertEqual(str(context.exception),
                         "cannot subtract Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            _ = tab1 - 5
        self.assertEqual(str(context.exception),
                         "cannot subtract Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 -= np.array([])
        self.assertEqual(str(context.exception),
                         "cannot subtract Tabulation in-place by given value")

        with self.assertRaises(ValueError) as context:
            tab1 -= 5
        self.assertEqual(str(context.exception),
                         "cannot subtract Tabulation in-place by given value")

        # MULTIPLICATION

        res = tab1 * tab2
        self.assertTrue(np.all(np.abs((tab1(tab1.x)*tab2(tab1.x) - res(tab1.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab1(tab2.x)*tab2(tab2.x) - res(tab2.x))) < 1e-10))

        res = tab2 * tab1
        self.assertTrue(np.all(np.abs((tab1(tab1.x)*tab2(tab1.x) - res(tab1.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab1(tab2.x)*tab2(tab2.x) - res(tab2.x))) < 1e-10))

        res = tab3 * tab4
        self.assertTrue(np.all(np.abs((tab3(tab3.x)*tab4(tab3.x) - res(tab3.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab3(tab4.x)*tab4(tab4.x) - res(tab4.x))) < 1e-10))

        res = tab4 * tab3
        self.assertTrue(np.all(np.abs((tab3(tab3.x)*tab4(tab3.x) - res(tab3.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab3(tab4.x)*tab4(tab4.x) - res(tab4.x))) < 1e-10))

        res = tab1 * tab3
        self.assertTrue(np.all(np.abs((tab1(tab1.x)*tab3(tab1.x) - res(tab1.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab1(tab3.x)*tab3(tab3.x) - res(tab3.x))) < 1e-10))

        res = tab3 * tab1
        self.assertTrue(np.all(np.abs((tab1(tab1.x)*tab3(tab1.x) - res(tab1.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab1(tab3.x)*tab3(tab3.x) - res(tab3.x))) < 1e-10))

        res = tab2 * tab4
        self.assertTrue(np.all(np.abs((tab2(tab2.x)*tab4(tab2.x) - res(tab2.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab2(tab4.x)*tab4(tab4.x) - res(tab4.x))) < 1e-10))

        res = tab4 * tab2
        self.assertTrue(np.all(np.abs((tab2(tab2.x)*tab4(tab2.x) - res(tab2.x))) < 1e-10))
        self.assertTrue(np.all(np.abs((tab2(tab4.x)*tab4(tab4.x) - res(tab4.x))) < 1e-10))

        res = tab1 * 10
        self.assertTrue(np.all(np.abs((tab1(off_xlist)*10 - res(off_xlist))) < 1e-10))

        tab5 = Tabulation(x1, y1)
        tab5 *= tab2
        self.assertTrue(np.all(np.abs((tab1(tab1.x)*tab2(tab1.x)-tab5(tab1.x))) < 1e-10))

        tab5 = Tabulation(x1, y1)
        tab5 *= 10
        self.assertTrue(np.all(np.abs((tab1(off_xlist)*10-tab5(off_xlist))) < 1e-10))

        with self.assertRaises(ValueError) as context:
            _ = tab1 * np.array([])
        self.assertEqual(str(context.exception),
                         "cannot multiply Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 *= np.array([])
        self.assertEqual(str(context.exception),
                         "cannot multiply Tabulation in-place by given value")

        # DIVISION

        res = tab1 / 10
        self.assertTrue(np.all(np.abs((tab1(off_xlist)/10 - res(off_xlist))) < 1e-10))

        tab5 = Tabulation(x1, y1)
        tab5 /= 10
        self.assertTrue(np.all(np.abs((tab1(off_xlist)/10 - tab5(off_xlist))) < 1e-10))

        with self.assertRaises(ValueError) as context:
            _ = tab1 / np.array([])
        self.assertEqual(str(context.exception),
                         "cannot divide Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 /= np.array([])
        self.assertEqual(str(context.exception),
                         "cannot divide Tabulation in-place by given value")

        # LOCATE

        for x in xs:
            self.assertEqual(tabs.locate(x)[0], x)
            self.assertEqual(len(tabs.locate(x)), 1)

        for x in xs[:-1]:
            self.assertEqual(tabs.locate(x+x/10)[0], x+x/10)
            self.assertEqual(len(tabs.locate(x+x/10)), 1)

        for x in xr[2:-3]:
            self.assertEqual(tabr.locate(x+x/10)[0], x+x/10)
            self.assertEqual(len(tabr.locate(x+x/10)), 2)

        self.assertEqual(tabr.locate(0.), [0, 11])
        self.assertEqual(tabr.locate(4.5), [4.5, 10.55])

        # CLIP

        clipped = tabr.clip(2, 5)
        self.assertEqual(clipped.domain(), (2., 5.))
        self.assertEqual(clipped.integral(), 10.5)

        clipped = tabr.clip(4.5, 5.5)
        self.assertEqual(clipped.domain(), (4.5, 5.5))
        self.assertEqual(clipped.integral(), 5.)

        clipped = tabr.clip(0, 12)
        self.assertEqual(clipped.domain(), (0, 11))
        self.assertEqual(clipped(0), 0)
        self.assertEqual(clipped(0.5), 0.5)
        self.assertEqual(clipped(11), 0)

        clipped = tabr.clip(-10, 20)
        self.assertEqual(clipped.domain(), (0, 11))
        self.assertEqual(clipped(0), 0)
        self.assertEqual(clipped(0.5), 0.5)
        self.assertEqual(clipped(11), 0)

        clipped = tabr.clip(0.5, 10)
        self.assertEqual(clipped.domain(), (0.5, 10))
        self.assertEqual(clipped(0.4), 0)
        self.assertEqual(clipped(0.5), 0.5)
        self.assertEqual(clipped(0.6), 0.6)
        self.assertEqual(clipped(10), 10)
        self.assertEqual(clipped(10.1), 0)
        self.assertEqual(clipped(11), 0)

        with self.assertRaises(ValueError) as context:
            tabr.clip(15, 16)
        self.assertEqual(str(context.exception), "domains do not overlap")

        # X_MEAN

        boxcar = Tabulation((0., 10.), (1., 1.))
        self.assertEqual(boxcar.x_mean(), 5.)

        self.assertTrue(np.abs(boxcar.x_mean(0.33) - 5.) < 1e-10)

        # BANDWIDTH_RMS
        self.assertTrue(np.abs(boxcar.bandwidth_rms() - 5.) < 1e-7)
        self.assertTrue(np.abs(boxcar.bandwidth_rms(0.001) - 5. / np.sqrt(3.)) < 1e-7)

        boxcar = Tabulation((10000, 10010), (1, 1))
        self.assertEqual(boxcar.x_mean(), 10005.)

        # PIVOT_MEAN

        # For narrow functions, the pivot_mean and the x_mean are similar
        self.assertTrue(np.abs(boxcar.pivot_mean(1.e-6) - 10005.) < 1e-3)

        # For broad functions, values differ
        boxcar = Tabulation((1, 100), (1, 1))
        self.assertTrue(np.abs(boxcar.pivot_mean(1.e-6) - 99. / np.log(100.)) < 1e-3)

        # FWHM

        triangle = Tabulation((0, 10, 20), (0, 1, 0))
        self.assertEqual(triangle.fwhm(), 10.)

        triangle = Tabulation((0, 10, 20), (0, 1, 0))
        self.assertEqual(triangle.fwhm(0.25), 15.)

        less_triangle = Tabulation((0, 10), (0, 1))
        with self.assertRaises(ValueError) as context:
            less_triangle.fwhm(0.5)
        self.assertEqual(str(context.exception),
                         "Tabulation does not cross fractional height twice")

        more_triangle = Tabulation((0, 10, 20, 30), (0, 1, 0, 1))
        with self.assertRaises(ValueError) as context:
            more_triangle.fwhm(0.5)
        self.assertEqual(str(context.exception),
                         "Tabulation does not cross fractional height twice")
        with self.assertRaises(ValueError) as context:
            more_triangle.fwhm(-0.0001)
        self.assertEqual(str(context.exception),
                         "fraction is outside the range 0-1")
        with self.assertRaises(ValueError) as context:
            more_triangle.fwhm(1.0001)
        self.assertEqual(str(context.exception),
                         "fraction is outside the range 0-1")

        # SQUARE_WIDTH

        self.assertEqual(triangle.square_width(), 10.)
        self.assertEqual(boxcar.square_width(), 99.)

        # INITIALIZATION ERRORS

        # Not 1 dimensional x
        x = np.array([[1, 2], [3, 4]])  # 2-dimensional array
        y = np.array([4, 5])
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception), "x array is not 1-dimensional")

        # Test initialization with x and y arrays of different sizes
        x = np.array([1, 2, 3])
        y = np.array([4, 5])  # Mismatched size
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception),
                         "x and y arrays do not have the same size")

        # Test initialization with a non-monotonic x array
        x = np.array([1, 3, 2])  # Non-monotonic
        y = np.array([4, 5, 6])
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception),
                         "x-coordinates are not strictly monotonic")

        # Test initialization with a non-monotonic x array (with floats)
        x = np.array([1., 3., 2.])  # Non-monotonic
        y = np.array([4., 5., 6.])
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception),
                         "x-coordinates are not strictly monotonic")

        # Test update with new_y having a different size than x
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        new_y = np.array([7, 8])  # Mismatched size
        with self.assertRaises(ValueError) as context:
            tab._update(tab.x, new_y)
        self.assertEqual(str(context.exception),
                         "x and y arrays do not have the same size")

        # Test xmerge with non-overlapping domains
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        with self.assertRaises(ValueError) as context:
            Tabulation._xmerge(x1, x2)
        self.assertEqual(str(context.exception), "domains do not overlap")

        # Test xmerge with non-overlapping domains (with floats)
        x1 = np.array([1., 2., 3.])
        x2 = np.array([4., 5., 6.])
        with self.assertRaises(ValueError) as context:
            Tabulation._xmerge(x1, x2)
        self.assertEqual(str(context.exception), "domains do not overlap")

        # QUANTILE, INTEGRATE with limits

        tab = Tabulation((0, 1), (0, 1))
        self.assertEqual(tab.quantile(0), 0)
        self.assertEqual(tab.quantile(1), 1)
        self.assertEqual(tab.quantile(0.25), 0.5)
        self.assertEqual(tab.quantile(0.0625), 0.25)
        self.assertEqual(tab.quantile(0.5625), 0.75)

        np.random.seed(3285)
        xrandom = np.cumsum(np.random.rand(100))
        yrandom = 0.1 * np.arange(100) + np.random.randn(100)
        yrandom[yrandom < 0] = 0.
        eps = 1.e-14
        for tab in (Tabulation(np.arange(8), np.ones(8)),
                    Tabulation(np.arange(9), np.ones(9)),
                    Tabulation(np.arange(8), np.arange(8)),
                    Tabulation(np.arange(9), np.arange(9)),
                    Tabulation(xrandom, yrandom)):
            integ = tab.integral()
            for q in (0, 1, 0.5, 1./3, 1./7, 3/7., 6/7., 0.001, 0.999, np.sqrt(0.5)):
                test1 = q * integ
                test2 = tab.integral(tab.x[0], tab.quantile(q))
                self.assertLessEqual(abs(test1 - test2), tab.x[-1] * eps)

        self.assertRaises(ValueError, tab.quantile, -1.e99)
        self.assertRaises(ValueError, tab.quantile, 1.0000000001)

        # GETITEM, STR, REPR

        tab = Tabulation(np.arange(9), np.arange(9))
        self.assertEqual(tab[4], tab.y[4])
        self.assertEqual(str(tab), 'Tabulation([0. 1. ... 7. 8.], [0. 1. ... 7. 8.])')
        self.assertEqual(str(tab), repr(tab))

        tab2 = tab[:4]
        self.assertEqual(tab2.domain()[1], tab.y[3])
        self.assertEqual(len(tab2), 4)
        self.assertEqual(str(tab2), 'Tabulation([0. 1. 2. 3.], [0. 1. 2. 3.])')
        self.assertEqual(str(tab2), repr(tab2))

        tab3 = tab[[0, 1, 3, 5, 8]]
        self.assertEqual(tab3.domain()[1], tab.y[8])
        self.assertEqual(len(tab3), 5)
        self.assertEqual(str(tab3), 'Tabulation([0. 1. ... 5. 8.], [0. 1. ... 5. 8.])')
        self.assertEqual(str(tab3), repr(tab3))

        tab4 = tab[1::2]
        self.assertEqual(tab4.domain()[0], tab.y[1])
        self.assertEqual(tab4.domain()[1], tab.y[7])
        self.assertEqual(len(tab4), 4)
        self.assertEqual(str(tab4), 'Tabulation([1. 3. 5. 7.], [1. 3. 5. 7.])')
        self.assertEqual(str(tab4), repr(tab4))

        tab5 = tab[np.array([True] + 7*[False] + [True])]
        self.assertEqual(tab5.domain()[0], tab.y[0])
        self.assertEqual(tab5.domain()[1], tab.y[-1])
        self.assertEqual(len(tab5), 2)
        self.assertEqual(str(tab5), 'Tabulation([0. 8.], [0. 8.])')
        self.assertEqual(str(tab5), repr(tab5))

        self.assertRaises(IndexError, tab.__getitem__, -99)
        self.assertRaises(ValueError, tab.__getitem__, None)
        self.assertRaises(ValueError, tab.__getitem__, [0, 1, 5, 4])

        # Delta function case

        tab = Tabulation([3], [7])
        self.assertEqual(tab.domain(), (3, 3))
        self.assertEqual(tab(3), 7)
        self.assertEqual(list(tab([2, 3, 4])), [0, 7, 0])
        self.assertEqual(tab(math.nextafter(3., -math.inf)), 0.)
        self.assertEqual(tab(math.nextafter(3.,  math.inf)), 0.)

        # All zeros

        tab = Tabulation(np.arange(10), np.zeros(10))
        self.assertEqual(tab.domain(), (0, 9))
        self.assertEqual(tab(3), 0)
        self.assertEqual(tab(-3), 0)
        self.assertEqual(list(tab([2, 3, 4])), [0, 0, 0])

        # Duplicated end points

        x = np.array([1., 1., 2., 3., 4., 4.])
        y = np.array([0., 4., 3., 2., 1., 0.])
        tab = Tabulation(x, y)
        self.assertEqual(tab.domain(), (1, 4))
        self.assertLess(tab.x[0], 1.)
        self.assertGreater(tab.x[-1], 4.)

        x = np.array([1., 1.1, 2., 3., 4., 4.])
        y = np.array([0., 4., 3., 2., 1., 0.])
        tab = Tabulation(x, y)
        self.assertEqual(tab.domain(), (1, 4))

        x = np.array([1., 1., 2., 3., 3.9, 4.])
        y = np.array([0., 4., 3., 2., 1., 0.])
        tab = Tabulation(x, y)
        self.assertEqual(tab.domain(), (1, 4))

        # Duplicated internal points

        tab = Tabulation([0, 1, 1, 2], [2, 4, 1, 3])
        self.assertEqual(tab.x[1], 1)
        self.assertEqual(tab.x[2], math.nextafter(1, np.inf))
        x = np.array([0.990, 0.995, 1, 1.005, 1.010])
        answer = np.array([3.98, 3.99, 4., 1.01, 1.02])
        diff = tab(x) - answer
        self.assertLess(np.abs(diff).max(), 1.e-15)
