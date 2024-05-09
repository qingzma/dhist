import unittest
import pandas as pd
import numpy as np

from joins.cnts.cnts import CumulativeDistinctCounter


class TestCounterMethod(unittest.TestCase):

    def test_parse_query_simple(self):
        counter = CumulativeDistinctCounter()
        data = pd.Series([1, 1, 1, 2, 2, 2, 5, 25, 1, 1, 3, 8, 8, 12])
        counter.fit(data)
        p0 = counter._predict(1)
        p1 = counter.predicts([1, 3, 10])
        self.assertEqual(p0, 1)
        self.assertEqual(list(p1), [1, 2, 3])

        p2 = counter.predicts([3, 12])
        self.assertEqual(list(p2), [3, 3])


if __name__ == "__main__":
    unittest.main()
