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
    NonKeyTopKHistogram,
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

    def test_NonKeyCumulativeHistogram_simple(self):

        data = []
        for i in reversed(range(1, 5)):
            # for i in range(1, 5):
            for _ in range(i):
                data.append(i)
        data = pd.DataFrame(data, columns=["a"])
        hist = NonKeyCumulativeHistogram(n_bins=20)
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

    def test_NonKeyCumulativeHistogram_large_categorical(self):

        data = []
        # for i in reversed(range(1, 120)):
        for i in range(1, 120):
            for _ in range(i):
                data.append(i)
        for i in range(10):
            data.append(117)
        data = pd.DataFrame(data, columns=["a"])
        # print(data)
        hist = NonKeyCumulativeHistogram(n_bins=50, n_categorical=150)
        hist.fit(data, headers=["a"])

        domain = Domain(mins=1, left=True, right=True)
        self.assertEqual(7150, hist.selectivity(domain, frac=False))
        domain = Domain(mins=0, left=True, right=True)
        self.assertEqual(7150, hist.selectivity(domain, frac=False))
        domain = Domain(mins=5, left=True, right=True)
        self.assertEqual(7135, hist.selectivity(domain, frac=False))

        # [1,], (1,)
        domain = Domain(mins=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7149) < 1e-5)
        domain = Domain(mins=1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)
        domain = Domain(mins=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)
        domain = Domain(mins=0.9, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)

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

        # [119,], (119,)
        domain = Domain(mins=119, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=119, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 119) < 1e-5)
        domain = Domain(mins=119.1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=119.1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)

    def test_NonKeyCumulativeHistogram_large_continuous(self):

        data = []
        # for i in reversed(range(1, 120)):
        for i in range(1, 120):
            for _ in range(i):
                data.append(i)
        for i in range(10):
            data.append(117)
        data = pd.DataFrame(data, columns=["a"])
        # print(data)
        hist = NonKeyCumulativeHistogram(n_bins=50, n_categorical=50)
        hist.fit(data, headers=["a"])

        error = 40
        domain = Domain(mins=1, left=True, right=True)
        self.assertTrue(abs(7150 - hist.selectivity(domain, frac=False)) < error)
        domain = Domain(mins=0, left=True, right=True)
        self.assertTrue(abs(7150 - hist.selectivity(domain, frac=False)) < error)
        domain = Domain(mins=5, left=True, right=True)
        self.assertTrue(abs(7135 - hist.selectivity(domain, frac=False)) < error)

        # [1,], (1,)
        domain = Domain(mins=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7149) < error)
        domain = Domain(mins=1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < error)
        domain = Domain(mins=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < error)
        domain = Domain(mins=0.9, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < error)

        # (,1), (,1]
        domain = Domain(maxs=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 1) < error)
        domain = Domain(maxs=1, left=True, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(maxs=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(maxs=0.9, left=True, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)

        # normal case
        domain = Domain(mins=1, left=True, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 6) < error)
        domain = Domain(mins=1, left=False, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 5) < error)
        domain = Domain(mins=1, left=True, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < error)
        domain = Domain(mins=1, left=False, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 9) < error)

        # [119,], (119,)
        domain = Domain(mins=119, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(mins=119, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 119) < error)
        domain = Domain(mins=119.1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(mins=119.1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)

    def test_NonKeyTopKHistogram_large_continuous(self):

        data = []
        # for i in reversed(range(1, 120)):
        for i in range(1, 120):
            for _ in range(i):
                data.append(i)
        for i in range(10):
            data.append(117)
        data = pd.DataFrame(data, columns=["a"])
        # print(data)
        hist = NonKeyTopKHistogram(n_top_k=5, n_bins=50, n_categorical=150)
        hist.fit(data, headers=["a"])

        domain = Domain(mins=1, left=True, right=True)
        self.assertEqual(7150, hist.selectivity(domain, frac=False))
        domain = Domain(mins=0, left=True, right=True)
        self.assertEqual(7150, hist.selectivity(domain, frac=False))
        domain = Domain(mins=5, left=True, right=True)
        self.assertEqual(7135, hist.selectivity(domain, frac=False))

        # [1,], (1,)
        domain = Domain(mins=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7149) < 1e-5)
        domain = Domain(mins=1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)
        domain = Domain(mins=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)
        domain = Domain(mins=0.9, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)

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
        domain = Domain(mins=1, left=False, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 9) < 1e-5)
        domain = Domain(mins=1, left=False, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 5) < 1e-5)
        domain = Domain(mins=1, left=True, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 6) < 1e-5)
        domain = Domain(mins=1, left=True, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < 1e-5)

        # [119,], (119,)
        domain = Domain(mins=119, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=119, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 119) < 1e-5)
        domain = Domain(mins=119.1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=119.1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)

    def test_NonKeyTopKHistogram_large_continuous(self):

        data = []
        for i in reversed(range(1, 120)):
            # for i in range(1, 120):
            for _ in range(i):
                data.append(i)
        for i in range(10):
            data.append(117)
        data = pd.DataFrame(data, columns=["a"])
        # print(data)
        hist = NonKeyTopKHistogram(n_top_k=5, n_bins=100, n_categorical=50)
        hist.fit(data, headers=["a"])

        domain = Domain(mins=1, left=True, right=True)
        self.assertEqual(7150, hist.selectivity(domain, frac=False))
        domain = Domain(mins=0, left=True, right=True)
        self.assertEqual(7150, hist.selectivity(domain, frac=False))
        domain = Domain(mins=5, left=False, right=True)
        self.assertEqual(7135, hist.selectivity(domain, frac=False))
        domain = Domain(mins=5, left=True, right=True)
        self.assertEqual(7140, hist.selectivity(domain, frac=False))

        # [1,], (1,)
        domain = Domain(mins=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7149) < 1e-5)
        domain = Domain(mins=1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)
        domain = Domain(mins=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)
        domain = Domain(mins=0.9, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < 1e-5)

        # (,5),(,5]
        domain = Domain(maxs=5, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 15) < 1e-5)
        domain = Domain(maxs=5, left=True, right=False)
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
        domain = Domain(mins=2, left=False, maxs=9, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 42) < 1e-5)
        domain = Domain(mins=2, left=False, maxs=6, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 18) < 1e-5)
        domain = Domain(mins=2, left=False, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7) < 1e-5)
        domain = Domain(mins=2, left=False, maxs=3, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 3) < 1e-5)
        domain = Domain(mins=1, left=False, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 5) < 1e-5)
        domain = Domain(mins=1, left=True, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 6) < 1e-5)
        domain = Domain(mins=1, left=True, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < 1e-5)

        # [119,], (119,)
        domain = Domain(mins=119, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=119, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 119) < 1e-5)
        domain = Domain(mins=119.1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)
        domain = Domain(mins=119.1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < 1e-5)

    def test_NonKeyTopKHistogram_large_continuous_less_bin(self):

        data = []
        for i in reversed(range(1, 120)):
            # for i in range(1, 120):
            for _ in range(i):
                data.append(i)
        for i in range(10):
            data.append(117)
        data = pd.DataFrame(data, columns=["a"])
        # print(data)
        hist = NonKeyTopKHistogram(n_top_k=5, n_bins=20, n_categorical=50)
        hist.fit(data, headers=["a"])

        error = 4
        domain = Domain(mins=1, left=True, right=True)
        self.assertEqual(7150, hist.selectivity(domain, frac=False))
        domain = Domain(mins=0, left=True, right=True)
        self.assertTrue(abs(7150 - hist.selectivity(domain, frac=False)) < error)
        domain = Domain(mins=5, left=False, right=True)
        self.assertTrue(abs(7135 - hist.selectivity(domain, frac=False)) < error)
        domain = Domain(mins=5, left=True, right=True)
        self.assertTrue(abs(7140 - hist.selectivity(domain, frac=False)) < error)

        # [1,], (1,)
        domain = Domain(mins=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7149) < error)
        domain = Domain(mins=1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < error)
        domain = Domain(mins=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < error)
        domain = Domain(mins=0.9, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7150) < error)

        # (,5),(,5]
        domain = Domain(maxs=5, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 15) < error)
        domain = Domain(maxs=5, left=True, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < error)

        # (,1), (,1]
        domain = Domain(maxs=1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 1) < error)
        domain = Domain(maxs=1, left=True, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(maxs=0.9, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(maxs=0.9, left=True, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)

        # normal case
        domain = Domain(mins=2, left=False, maxs=9, right=True)
        # print("prediction ", hist.selectivity(domain, frac=False))
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 42) < error)
        domain = Domain(mins=2, left=False, maxs=6, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 18) < error)
        domain = Domain(mins=2, left=False, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 7) < error)
        domain = Domain(mins=2, left=False, maxs=3, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 3) < error)
        domain = Domain(mins=1, left=False, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 5) < error)
        domain = Domain(mins=1, left=True, maxs=4, right=False)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 6) < error)
        domain = Domain(mins=1, left=True, maxs=4, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 10) < error)

        # [119,], (119,)
        domain = Domain(mins=119, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(mins=119, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 119) < error)
        domain = Domain(mins=119.1, left=False, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)
        domain = Domain(mins=119.1, left=True, right=True)
        self.assertTrue(abs(hist.selectivity(domain, frac=False) - 0) < error)


if __name__ == "__main__":
    unittest.main()
