import unittest
from joins.pdf.kde import Kde
import numpy as np


class TestPdfMethod(unittest.TestCase):

    def test_upper(self):
        X = np.array([[1], [2], [3], [1], [2], [2]])
        pdf = Kde()
        pdf.fit(X, kernel='linear')
        dens = pdf.predict(np.array([[1]]))
        self.assertTrue((dens-0.28918017) < 1e-3)


if __name__ == '__main__':
    unittest.main()
