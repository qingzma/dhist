import pandas as pd
import numpy as np
from bisect import bisect_left


class CumulativeDistinctCounter:
    def __init__(self) -> None:
        self.uniques = None
        self.uniques_counts = None
        # self.counts = None

    def fit(self, data: np.ndarray):
        x = data.to_numpy().reshape(-1, 1)
        self.uniques, self.counts = np.unique(x, return_counts=True)
        self.uniques_counts = np.cumsum(np.ones_like(self.uniques))
        print(self.uniques)
        print(self.uniques_counts)
        # print(self.counts)

    def _predict(self, point):
        return bisect_left(self.uniques, point) + 1

    def predicts(self, points):
        cnts = [self._predict(point) for point in points]
        # print("cnts: ", cnts)
        minused = [y - x for x, y in zip(cnts, cnts[1:])]
        minused.insert(0, cnts[0])
        return np.array(minused)
