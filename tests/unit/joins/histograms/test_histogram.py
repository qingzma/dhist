import unittest

import numpy as np
import pandas as pd

from joins.histograms.histogram1d import (
    JoinHistogram,
    TableJoin,
    UpperBoundHistogram,
    UpperBoundHistogramTopK,
)


class TestHistogramMethod(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.b = pd.read_csv("data/stats/badges.csv")[["UserId"]]
        self.c = pd.read_csv("data/stats/comments.csv")[["UserId"]]
        self.u = pd.read_csv("data/stats/users.csv")[["Id"]]
        self.ph = pd.read_csv("data/stats/postHistory.csv")[["UserId"]]
        self.p = pd.read_csv("data/stats/posts.csv")[["OwnerUserId"]]

        low = np.min(
            [
                self.b.min().values[0],
                self.c.min().values[0],
                self.u.min().values[0],
                self.ph.min().values[0],
                self.p.min().values[0],
            ]
        )
        high = np.max(
            [
                self.b.max().values[0],
                self.c.max().values[0],
                self.u.max().values[0],
                self.ph.max().values[0],
                self.p.max().values[0],
            ]
        )
        self.bins = np.linspace(low, high, 300)

    # def test_table_join(self):
    #     tj_b = TableJoin()
    #     tj_b.fit(self.b, ["UserId"])
    #     tj_c = TableJoin()
    #     tj_c.fit(self.c, ["UserId"])
    #     tj = tj_b.join(tj_c, bins=self.bins)
    #     self.assertEqual(np.sum(tj), 15900001)

    # def test_join_histogram(self):
    #     jh_b = JoinHistogram()
    #     jh_b.fit(self.b, ["UserId"], self.bins)
    #     jh_c = JoinHistogram()
    #     jh_c.fit(self.c, ["UserId"], self.bins)
    #     jh = jh_b.join(jh_c)
    #     self.assertTrue(abs(15900001 - np.sum(jh)) / 15900001.0 < 0.98)

    # def test_upper_bound_histogram(self):
    #     ub_b = UpperBoundHistogram()
    #     ub_b.fit(self.b, ["UserId"], self.bins)
    #     ub_c = UpperBoundHistogram()
    #     ub_c.fit(self.c, ["UserId"], self.bins)
    #     ub = ub_b.join(ub_c)
    #     self.assertTrue(abs(15900001 - np.sum(ub)) / 15900001.0 < 1)

    def test_upper_bound_histogram_top_k(self):
        ubtk_b = UpperBoundHistogramTopK(5)
        ubtk_b.fit(self.b, ["UserId"], self.bins)
        ubtk_c = UpperBoundHistogramTopK(5)
        ubtk_c.fit(self.c, ["UserId"], self.bins)
        ubtk_u = UpperBoundHistogramTopK(5)
        ubtk_u.fit(self.u, ["Id"], self.bins)
        ubtk_p = UpperBoundHistogramTopK(5)
        ubtk_p.fit(self.p, ["OwnerUserId"], self.bins)
        # ubtk = ubtk_b.join(ubtk_c)
        # self.assertTrue(abs(15900001 - np.sum(ubtk)) / 15900001.0 < 0.1)
        # ubtk = ubtk_p.join(ubtk_b, update_statistics=True).join(ubtk_c)
        # self.assertTrue(abs(15131840763 - np.sum(ubtk)) / 15131840763.0 < 0.1)

        ubtk = ubtk_p.join(ubtk_b, update_statistics=True).join(ubtk_u)
        self.assertTrue(abs(3728360 - np.sum(ubtk)) / 3728360.0 < 0.1)


if __name__ == "__main__":
    unittest.main()
