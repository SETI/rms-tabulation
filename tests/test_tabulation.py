########################################
# UNIT TESTS
########################################

from tabulation import Tabulation

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
        yr = [0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 0., 0.]

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
        self.assertEqual(str(context.exception), "Cannot add Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            _ = tab1 + 5
        self.assertEqual(str(context.exception), "Cannot add Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 += np.array([])
        self.assertEqual(str(context.exception),
                         "Cannot add Tabulation in-place by given value")

        with self.assertRaises(ValueError) as context:
            tab1 += 5
        self.assertEqual(str(context.exception),
                         "Cannot add Tabulation in-place by given value")

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
                         "Cannot subtract Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            _ = tab1 - 5
        self.assertEqual(str(context.exception),
                         "Cannot subtract Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 -= np.array([])
        self.assertEqual(str(context.exception),
                         "Cannot subtract Tabulation in-place by given value")

        with self.assertRaises(ValueError) as context:
            tab1 -= 5
        self.assertEqual(str(context.exception),
                         "Cannot subtract Tabulation in-place by given value")

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
                         "Cannot multiply Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 *= np.array([])
        self.assertEqual(str(context.exception),
                         "Cannot multiply Tabulation in-place by given value")

        # DIVISION

        res = tab1 / 10
        self.assertTrue(np.all(np.abs((tab1(off_xlist)/10 - res(off_xlist))) < 1e-10))

        tab5 = Tabulation(x1, y1)
        tab5 /= 10
        self.assertTrue(np.all(np.abs((tab1(off_xlist)/10 - tab5(off_xlist))) < 1e-10))

        with self.assertRaises(ValueError) as context:
            _ = tab1 / np.array([])
        self.assertEqual(str(context.exception),
                         "Cannot divide Tabulation by given value")

        with self.assertRaises(ValueError) as context:
            tab1 /= np.array([])
        self.assertEqual(str(context.exception),
                         "Cannot divide Tabulation in-place by given value")

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

        with self.assertRaises(ValueError) as context:
            tabr.clip(15, 16)
        self.assertEqual(str(context.exception), "Domains do not overlap")

        # X_MEAN

        boxcar = Tabulation((0., 10.), (1., 1.))
        self.assertEqual(boxcar.x_mean(), 5.)

        self.assertTrue(np.abs(boxcar.x_mean(0.33) - 5.) < 1e-10)

        # BANDWIDTH_RMS
        self.assertTrue(np.abs(boxcar.bandwidth_rms() - 5.) < 1e-7)
        self.assertTrue(np.abs(boxcar.bandwidth_rms(0.001) - 5. / np.sqrt(3.)) < 1e-7)

        boxcar = Tabulation((10000, 10010),(1, 1))
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
        self.assertEqual(str(context.exception), "x-coordinates are not monotonic")

        # Test initialization with a non-monotonic x array (with floats)
        x = np.array([1., 3., 2.])  # Non-monotonic
        y = np.array([4., 5., 6.])
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception), "x-coordinates are not monotonic")

        # Test update with new_y having a different size than x
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        new_y = np.array([7, 8])  # Mismatched size
        with self.assertRaises(ValueError) as context:
            tab._update_y(new_y)
        self.assertEqual(str(context.exception),
                         "x and y arrays do not have the same size")

        # Test xmerge with non-overlapping domains
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        with self.assertRaises(ValueError) as context:
            Tabulation._xmerge(x1, x2)
        self.assertEqual(str(context.exception), "Domains do not overlap")

        # Test xmerge with non-overlapping domains (with floats)
        x1 = np.array([1., 2., 3.])
        x2 = np.array([4., 5., 6.])
        with self.assertRaises(ValueError) as context:
            Tabulation._xmerge(x1, x2)
        self.assertEqual(str(context.exception), "Domains do not overlap")
