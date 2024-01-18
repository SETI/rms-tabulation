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

        tab = Tabulation(x,y)

        self.assertEqual(4., tab(4))
        self.assertEqual(4.5, tab(4.5))
        self.assertEqual(0., tab(10.000000001))

        self.assertEqual(tab.domain(), (0.,10.))

        reversed = Tabulation(x[::-1],y)
        self.assertEqual(4., reversed(6))
        self.assertEqual(4.5, reversed(5.5))
        self.assertEqual(0., reversed(10.000000001))

        self.assertTrue(np.all(np.array((3.5,4.5,5.5)) == tab((3.5,4.5,5.5))))
        self.assertTrue(tab.integral(), 50.)

        resampled = tab.resample(np.arange(0,10.5,0.5))
        self.assertTrue(np.all(resampled.y == resampled.x))

        resampled = tab.resample(np.array((0.,10.)))
        self.assertTrue(np.all(resampled.y == resampled.x))

        xlist = np.arange(0.,10.25,0.25)
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

        clipped = resampled.clip(-5,5)
        self.assertEqual(clipped.domain(), (-5.,5.))
        self.assertEqual(clipped.integral(), 12.5)

        clipped = resampled.clip(4.5,5.5)
        self.assertEqual(clipped.domain(), (4.5,5.5))
        self.assertEqual(clipped.integral(), 5.)

        ratio = tab / clipped
        self.assertEqual(ratio.domain(), (4.5,5.5))
        self.assertEqual(ratio(4.49999), 0.)
        self.assertEqual(ratio(4.5), 1.)
        self.assertEqual(ratio(5.1), 1.)
        self.assertEqual(ratio(5.5), 1.)
        self.assertEqual(ratio(5.500001), 0.)

        product = ratio * clipped
        self.assertEqual(product.domain(), (4.5,5.5))
        self.assertEqual(product(4.49999), 0.)
        self.assertEqual(product(4.5), 4.5)
        self.assertEqual(product(5.1), 5.1)
        self.assertEqual(product(5.5), 5.5)
        self.assertEqual(product(5.500001), 0.)

        # mean()
        boxcar = Tabulation((0.,10.),(1.,1.))
        self.assertEqual(boxcar.mean(), 5.)

        eps = 1.e-14
        self.assertTrue(np.abs(boxcar.mean(0.33) - 5.) < eps)

        # bandwidth_rms()
        value = 5. / np.sqrt(3.)
        eps = 1.e-7
        self.assertTrue(np.abs(boxcar.bandwidth_rms(0.001) - value) < eps)

        boxcar = Tabulation((10000,10010),(1,1))
        self.assertEqual(boxcar.mean(), 10005.)

        # pivot_mean()
        # For narrow functions, the pivot_mean and the mean are similar
        eps = 1.e-3
        self.assertTrue(np.abs(boxcar.pivot_mean(1.e-6) - 10005.) < eps)

        # For broad functions, values differ
        boxcar = Tabulation((1,100),(1,1))
        value = 99. / np.log(100.)
        eps = 1.e-3
        self.assertTrue(np.abs(boxcar.pivot_mean(1.e-6) - value) < eps)

        # fwhm()
        triangle = Tabulation((0,10,20),(0,1,0))
        self.assertEqual(triangle.fwhm(), 10.)

        triangle = Tabulation((0,10,20),(0,1,0))
        self.assertEqual(triangle.fwhm(0.25), 15.)

        # square_width()
        self.assertEqual(triangle.square_width(), 10.)
        self.assertEqual(boxcar.square_width(), 99.)


        # Not 1 dimensional x
        x = np.array([[1, 2], [3, 4]])  # 2-dimensional array
        y = np.array([4, 5])
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception), "x array in not 1-dimensional")

        # Test initialization with x and y arrays of different sizes
        x = np.array([1, 2, 3])
        y = np.array([4, 5])  # Mismatched size
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception), "x and y arrays do not have the same size")

        # Test initialization with a non-monotonic x array
        x = np.array([1, 3, 2])  # Non-monotonic
        y = np.array([4, 5, 6])
        with self.assertRaises(ValueError) as context:
            Tabulation(x, y)
        self.assertEqual(str(context.exception), "x-coordinates are not monotonic")

        # Test update with new_y set to None
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        result = tab._update_y(None)
        self.assertIs(result, tab)  # Should return the original object

        # Test update with new_y having a different size than x
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        new_y = np.array([7, 8])  # Mismatched size
        with self.assertRaises(ValueError) as context:
            tab._update_y(new_y)
        self.assertEqual(str(context.exception), "x and y arrays do not have the same size")

        # Test xmerge with non-overlapping domains
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        with self.assertRaises(ValueError) as context:
            result = Tabulation._xmerge(x1, x2)
        self.assertEqual(str(context.exception), "domains do not overlap")

        # Test in-place multiplication of two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([2, 3, 4])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        tab1 *= tab2
        expected_x = np.array([2, 3])  # Intersection of x1 and x2
        expected_y = y1[1:] * y2[:-1]
        self.assertTrue(np.array_equal(tab1.x, expected_x))
        self.assertTrue(np.array_equal(tab1.y, expected_y))

        # # Test in-place multiplication with a scalar
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        tab = Tabulation(x, y)
        scalar_value = 2.0

        tab *= scalar_value
        expected_y = y * scalar_value
        self.assertTrue(np.array_equal(tab.y, expected_y))


        # Test in-place division of two Tabulations
        x1 = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        tab1 = Tabulation(x1, y1)

        x2 = np.array([2, 3, 4])
        y2 = np.array([1, 2, 3])
        tab2 = Tabulation(x2, y2)

        result = tab1.__idiv__(tab2)

        expected_x = np.array([2, 3])  # Intersection of x1 and x2
        expected_y = y1[1:] / y2[:-1]

        self.assertTrue(np.array_equal(result.x, expected_x))
        self.assertTrue(np.array_equal(result.y, expected_y))


        # Test subtraction of two Tabulations
        # x1 = np.array([1, 2, 3])
        # y1 = np.array([4, 5, 6])
        # tab1 = Tabulation(x1, y1)

        # x2 = np.array([2, 3, 4])
        # y2 = np.array([1, 2, 3])
        # tab2 = Tabulation(x2, y2)

        # result = tab1.__sub__(tab2)

        # expected_x = np.array([1., 2., 3., 4.])  # Merged x values (float)
        # expected_y = np.array([3, 3, 3, -3])  # Result of subtraction

        # self.assertTrue(np.array_equal(result.x, expected_x))
        # self.assertTrue(np.array_equal(result.y, expected_y))