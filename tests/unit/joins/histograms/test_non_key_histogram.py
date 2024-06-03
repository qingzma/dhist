import unittest

import numpy as np
import pandas as pd

from joins.domain import Domain
from joins.histograms.histograms import (  # UpperBoundHistogramTopK2D,
    JoinHistogram,
    TableJoin,
    UpperBoundHistogram,
    UpperBoundHistogramTopK,
)
from joins.histograms.non_key_histogram import (
    NonKeyCumulativeHistogram,
    NonKeyHistogram,
)


class TestHistogramMethod(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        # self.b = pd.read_csv("data/stats/badges.csv")[["UserId"]]
        # self.c = pd.read_csv("data/stats/comments.csv")[["UserId"]]
        # self.u = pd.read_csv("data/stats/users.csv")[["Id"]]
        # self.ph = pd.read_csv("data/stats/postHistory.csv")[["UserId"]]
        # self.p = pd.read_csv("data/stats/posts.csv")[["OwnerUserId"]]
        # self.p2d = pd.read_csv("data/stats/posts.csv")[["OwnerUserId", "Id"]]
        # self.pl = pd.read_csv("data/stats/postLinks.csv")[["PostId"]]

        # low = np.min(
        #     [
        #         self.b.min().values[0],
        #         self.c.min().values[0],
        #         self.u.min().values[0],
        #         self.ph.min().values[0],
        #         self.p.min().values[0],
        #     ]
        # )
        # high = np.max(
        #     [
        #         self.b.max().values[0],
        #         self.c.max().values[0],
        #         self.u.max().values[0],
        #         self.ph.max().values[0],
        #         self.p.max().values[0],
        #     ]
        # )
        # self.bins = np.linspace(low, high, 200)

        # low1 = np.min([self.p2d.min().values[1], self.pl.min().values[0]])
        # high1 = np.max([self.p2d.max().values[1], self.pl.max().values[0]])

        # self.bins1 = np.linspace(low1, high1, 200)
        # self.grid_x, self.grid_y = np.meshgrid(self.bins, self.bins1)
        # self.grid = np.array([self.grid_x.flatten(), self.grid_y.flatten()]).T

    def test_NonKeyCumulativeHistogram(self):

        data = []
        for i in reversed(range(1, 5)):
            # for i in range(1, 5):
            for _ in range(i):
                data.append(i)
        data = pd.DataFrame(data, columns=["a"])
        hist = NonKeyCumulativeHistogram(n_top_k=3, n_total=20)
        hist.fit(data, headers=["a"])

        domain = Domain(mins=1, left=True, right=True)
        self.assertEqual(10, hist.selectivity(domain, frac=False))
        domain = Domain(mins=0, left=True, right=True)
        self.assertEqual(10, hist.selectivity(domain, frac=False))
        domain = Domain(mins=6, left=True, right=True)
        self.assertEqual(0, hist.selectivity(domain, frac=False))

        # [1,], (1,)
        domain = Domain(mins=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 9) < 1e-5)
        domain = Domain(mins=1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < 1e-5)
        domain = Domain(mins=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < 1e-5)
        domain = Domain(mins=0.9, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < 1e-5)

        # (,1), (,1]
        domain = Domain(maxs=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 1) < 1e-5)
        domain = Domain(maxs=1, left=True, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(maxs=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(maxs=0.9, left=True, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)

        # normal case
        domain = Domain(mins=1, left=True, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 6) < 1e-5)
        domain = Domain(mins=1, left=False, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 5) < 1e-5)
        domain = Domain(mins=1, left=True, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < 1e-5)
        domain = Domain(mins=1, left=False, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 9) < 1e-5)

        # [4,], (4,)
        domain = Domain(mins=4, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=4, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 4) < 1e-5)
        domain = Domain(mins=4.1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=4.1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)


if __name__ == "__main__":
    unittest.main()
