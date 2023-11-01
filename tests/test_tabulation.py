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
