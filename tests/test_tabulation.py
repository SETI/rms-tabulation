########################################
# UNIT TESTS
########################################

from tabulation import Tabulation

import numpy as np
import unittest


class Test_Tabulation(unittest.TestCase):

    def runTest(self):

        x = np.arange(11)
        y = np.arange(11)

        tab = Tabulation(x, y)

        self.assertEqual(4., tab(4))
        self.assertEqual(4.5, tab(4.5))
        self.assertEqual(0., tab(10.000000001))

        self.assertEqual(tab.domain(), (0., 10.))

        reversed = Tabulation(x[::-1], y)
        self.assertEqual(4., reversed(6))
        self.assertEqual(4.5, reversed(5.5))
        self.assertEqual(0., reversed(10.000000001))

        self.assertTrue(np.all(np.array((3.5, 4.5, 5.5)) == tab((3.5, 4.5, 5.5))))
        self.assertTrue(tab.integral(), 50.)

        resampled = tab.resample(np.arange(0, 10.5, 0.5))
        self.assertTrue(np.all(resampled.y == resampled.x))

        resampled = tab.resample(np.array((0., 10.)))
        self.assertTrue(np.all(resampled.y == resampled.x))

        xlist = np.arange(0., 10.25, 0.25)
        self.assertTrue(np.all(xlist == resampled(xlist)))
        self.assertTrue(np.all(xlist == tab(xlist)))

        sum = tab + reversed
        self.assertTrue(np.all(sum.y == 10.))

        sum = tab + 10.
        self.assertTrue(np.all(sum(xlist) - tab(xlist) == 10.))

        diff = sum - 10.
        self.assertTrue(np.all(diff(xlist) - tab(xlist) == 0.))

        scaled = tab * 2.
        self.assertTrue(np.all(scaled(xlist)/2. == tab(xlist)))

        rescaled = scaled / 2.
        self.assertTrue(np.all(rescaled(xlist) == tab(xlist)))
        self.assertTrue(np.all(rescaled(xlist) == resampled(xlist)))

        for x in xlist:
            self.assertEqual(tab.locate(x)[0], x)
            self.assertEqual(len(tab.locate(x)), 1)

        clipped = resampled.clip(-5, 5)
        self.assertEqual(clipped.domain(), (-5., 5.))
        self.assertEqual(clipped.integral(), 12.5)

        clipped = resampled.clip(4.5, 5.5)
        self.assertEqual(clipped.domain(), (4.5, 5.5))
        self.assertEqual(clipped.integral(), 5.)

        ratio = tab / clipped
        self.assertEqual(ratio.domain(), (4.5, 5.5))
        self.assertEqual(ratio(4.49999), 0.)
        self.assertEqual(ratio(4.5), 1.)
        self.assertEqual(ratio(5.1), 1.)
        self.assertEqual(ratio(5.5), 1.)
        self.assertEqual(ratio(5.500001), 0.)

        product = ratio * clipped
        self.assertEqual(product.domain(), (4.5, 5.5))
        self.assertEqual(product(4.49999), 0.)
        self.assertEqual(product(4.5), 4.5)
        self.assertEqual(product(5.1), 5.1)
        self.assertEqual(product(5.5), 5.5)
        self.assertEqual(product(5.500001), 0.)

        # mean()
        boxcar = Tabulation((0., 10.), (1., 1.))
        self.assertEqual(boxcar.mean(), 1.)

        # bandwidth_rms()
        # value = 5. / np.sqrt(3.)
        # eps = 1.e-7
        # self.assertTrue(np.abs(boxcar.bandwidth_rms() - value) < eps)

        # pivot_mean()
        # For narrow functions, the pivot_mean and the mean are similar
        # eps = 1.e-3
        # print(boxcar.pivot_mean(1.e-6))
        # self.assertTrue(np.abs(boxcar.pivot_mean(1.e-6) - 1.) < eps)

        # # For broad functions, values differ
        # boxcar = Tabulation((1, 100), (1, 1))
        # value = 99. / np.log(100.)
        # eps = 1.e-3
        # self.assertTrue(np.abs(boxcar.pivot_mean(1.e-6) - value) < eps)

        # fwhm()
        triangle = Tabulation((0, 10, 20), (0, 1, 0))
        self.assertEqual(triangle.fwhm(), 10.)

        triangle = Tabulation((0, 10, 20), (0, 1, 0))
        self.assertEqual(triangle.fwhm(0.25), 15.)

        # square_width()
        self.assertEqual(triangle.square_width(), 10.)
        self.assertEqual(boxcar.square_width(), 10.)

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
        self.assertEqual(
            str(context.exception), "x-coordinates are not monotonic")

        # Test initialization with a non-monotonic x array (with floats)
        x = np.array([1., 3., 2.])  # Non-monotonic
        y = np.array([4., 5., 6.])
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(
            str(context.exception), "x-coordinates are not monotonic")

        # Test update with new_y having a different size than x
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        new_y = np.array([7, 8])  # Mismatched size
        with self.assertRaises(ValueError) as context:
            tab._update_y(new_y)
        self.assertEqual(
            str(context.exception),
            "x and y arrays do not have the same size"
            )

        # Test xmerge with non-overlapping domains
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        with self.assertRaises(ValueError) as context:
            result = Tabulation._xmerge(x1, x2)
        self.assertEqual(str(context.exception), "domains do not overlap")

        # Test xmerge with non-overlapping domains (with floats)
        x1 = np.array([1., 2., 3.])
        x2 = np.array([4., 5., 6.])
        with self.assertRaises(ValueError) as context:
            result = Tabulation._xmerge(x1, x2)
        self.assertEqual(str(context.exception), "domains do not overlap")

        # resample where x=None
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        tab = Tabulation(x, y)

        resampled = tab.resample(None)

        self.assertTrue(np.all(resampled.x == x))
        self.assertTrue(np.all(resampled.y == y))

        # bandwidth_rms with dx=None
        # boxcar = Tabulation((0., 10.), (1., 1.))
        # value = 5

        # self.assertTrue(np.abs(boxcar.bandwidth_rms() - value) == 0.)

        # Test multiplication of Two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([1, 2, 3])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        result = tab1 * tab2

        expected_x = [1, 2, 3]  # Merged x values

        self.assertTrue(np.array_equal(result.x, expected_x))
        self.assertTrue(np.array_equal(result(x), tab1(x) * tab2(x)))
        self.assertTrue(np.array_equal(result.y, tab1.y * tab2.y))
        self.assertTrue(np.array_equal(result.y, y1 * y2))

        # Test multiplication of Two Tabulations with a scalar
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        scalar = 2

        result = tab * scalar

        assert np.array_equal(result(x), tab(x) * scalar)
        assert np.array_equal(result.y, tab.y * scalar)
        assert np.array_equal(result.y, y * scalar)

        # Test multiplication of Two Tabulations with a scalar (with floats)
        x = np.array([1., 2., 3.])
        y = np.array([4., 5., 6.])
        tab = Tabulation(x, y)

        scalar = 2.0

        result = tab * scalar

        assert np.array_equal(result(x), tab(x) * scalar)
        assert np.array_equal(result.y, tab.y * scalar)
        assert np.array_equal(result.y, y * scalar)

        # Testing multiplication with bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            result = tab * bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot multiply Tabulation by given value"
            )

        # Test in-place multiplication of two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([1, 2, 3])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        tab1 *= tab2
        expected_x = np.array([1, 2, 3])  # Intersection of x1 and x2
        expected_y = [4., 10., 18.]

        assert np.array_equal(expected_x, tab1.x)
        assert np.array_equal(expected_y, tab1.y)
        assert np.array_equal(expected_y, tab1(x1))

        # Test in-place multiplication with a scalar
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        scalar_value = 2.0

        tab *= scalar_value
        expected_y = y * scalar_value
        self.assertTrue(np.array_equal(expected_y, tab.y))
        self.assertTrue(np.array_equal(expected_y, tab(x)))

        # Testing in-place multiplication with bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            tab *= bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot multiply Tabulation in-place by given value"
            )

        # Test in-place division of two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([1, 2, 3])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        tab1 /= tab2

        expected_x = np.array([1., 2., 3.])  # Intersection of x1 and x2
        expected_y = np.array([4., 2.5, 2.])

        self.assertTrue(np.array_equal(expected_x, tab1.x))
        self.assertTrue(np.array_equal(expected_y, tab1.y))
        self.assertTrue(np.array_equal(expected_y, tab1(x1)))

        # Test in-place division of a Tabulations with a scalar
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        scalar_value = 2

        tab /= scalar_value
        expected_y = y / scalar_value
        self.assertTrue(np.array_equal(expected_y, tab.y))
        self.assertTrue(np.array_equal(expected_y, tab(x)))

        # Test in-place division of a Tabulations with a scalar (with floats)
        x = np.array([1., 2., 3.])
        y = np.array([4., 5., 6.])
        tab = Tabulation(x, y)
        scalar_value = 2.0

        tab /= scalar_value
        expected_y = y / scalar_value
        self.assertTrue(np.array_equal(expected_y, tab.y))
        self.assertTrue(np.array_equal(expected_y, tab(x)))

        # Testing division with a bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            result = tab / bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot divide Tabulation by given value"
            )

        # Testing in-place division with a bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            tab /= bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot divide Tabulation in-place by given value"
            )

        # Test addition of two Tabulations with a scalar
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        scalar = 2

        tab += scalar

        self.assertTrue(np.array_equal(tab.x, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(tab.y, np.array([4, 5, 6]) + scalar))
        self.assertTrue(np.array_equal(tab(x), np.array([4, 5, 6]) + scalar))
        self.assertTrue(np.array_equal(y + scalar, np.array([4, 5, 6]) + scalar))

        # Test addition of two Tabulations with a scalar (with floats)
        x = np.array([1., 2., 3.])
        y = np.array([4., 5., 6.])
        tab = Tabulation(x, y)

        scalar = 2.0

        tab += scalar

        self.assertTrue(np.array_equal(tab.x, np.array([1., 2., 3.])))
        self.assertTrue(np.array_equal(tab.y, np.array([4., 5., 6.]) + scalar))
        self.assertTrue(np.array_equal(tab(x), np.array([4., 5., 6.]) + scalar))
        self.assertTrue(np.array_equal(y + scalar, np.array([4., 5., 6.]) + scalar))

        # Test in-place addition of two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([1, 2, 3])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        tab1 += tab2
        expected_x = np.array([1, 2, 3])  # Merge of x1 and x2
        expected_y = np.array([6, 9, 12])

        self.assertTrue(np.array_equal(expected_x, tab1.x))
        self.assertTrue(np.array_equal(expected_y, tab1.y + tab2.y))
        self.assertTrue(np.array_equal(expected_y, tab1(x1) + tab2(x2)))

        # Testing in-place addition with a bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            tab += bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot add Tabulation in-place by given value"
            )

        # Testing addition with a bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            result = tab + bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot add Tabulation by given value"
            )

        # Test subtraction of two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([1, 2, 3])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        result = tab1 - tab2

        expected_x = [1, 2, 3]  # Merged x values

        self.assertTrue(np.array_equal(result.x, expected_x))
        self.assertTrue(np.array_equal(result.y, tab1.y - tab2.y))
        self.assertTrue(np.array_equal(result.y, y1 - y2))
        self.assertTrue(np.array_equal(result(x1), tab1(x1) - tab2(x2)))

        # Testing subtraction with a bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            result = tab - bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot subtract Tabulation by given value"
            )

        # Test in-place subtraction of two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([1, 2, 3])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        tab1 -= tab2
        expected_x = np.array([1, 2, 3])  # Merge of x1 and x2
        expected_y = np.array([2, 1, 0])

        self.assertTrue(np.array_equal(expected_x, tab1.x))
        self.assertTrue(np.array_equal(expected_y, tab1.y - tab2.y))
        self.assertTrue(np.array_equal(expected_y, tab1(x1) - tab2(x2)))

        # Test in-place subtraction of two Tabulations with a scalar
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        scalar = 2

        tab -= scalar

        self.assertTrue(np.array_equal(tab.x, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(tab.y, np.array([4, 5, 6]) - scalar))
        self.assertTrue(np.array_equal(tab(x), np.array([4, 5, 6]) - scalar))

        # Test in-place subtraction of two Tabulations with a scalar (with floats)
        x = np.array([1., 2., 3.])
        y = np.array([4., 5., 6.])
        tab = Tabulation(x, y)

        scalar = 2.0

        tab -= scalar

        self.assertTrue(np.array_equal(tab.x, np.array([1., 2., 3.])))
        self.assertTrue(np.array_equal(tab.y, np.array([4., 5., 6.]) - scalar))
        self.assertTrue(np.array_equal(tab(x), np.array([4., 5., 6.]) - scalar))

        # Testing subtraction in-place with a bad array
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)

        bad_array = np.array([])
        with self.assertRaises(ValueError) as context:
            tab -= bad_array
        self.assertEqual(
            str(context.exception),
            "Cannot subtract Tabulation in-place by given value"
            )
