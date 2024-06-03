from bisect import bisect

import numpy as np
import pandas as pd

from joins.domain import Domain
from joins.tools import division


def interp(x, y, point):
    # print("x is ", x)
    # print("y is ", y)
    # print("point is ", point)
    return (y[1] - y[0]) * (point - x[0]) / (x[1] - x[0]) + y[0]


class NonKeyCumulativeHistogram:
    def __init__(self, n_top_k=30, n_total=100, n_categorical=300) -> None:
        assert n_total >= n_top_k
        self.n_top_k = n_top_k
        self.n_total = n_total
        self.n_categorical = n_categorical
        self.bins = None
        self.is_categorical = False
        self.cdf = None
        self.size = None
        self.bin_width = 1.0

    def fit(self, data: pd.DataFrame, headers: list) -> None:
        assert len(headers) == 1
        self.size = data.shape[0]
        uniques = np.sort(pd.unique(data[headers[0]]))
        if len(uniques) < self.n_categorical:
            self.is_categorical = True
            self.bins = uniques
        else:
            self.bins = np.linspace(uniques[0], uniques[-1], self.n_total)
            self.bin_width = self.bins[1] - self.bins[0]

        data["count"] = pd.cut(data[headers[0]], bins=self.bins, labels=self.bins[:-1])
        # print(data)
        counts = (
            data.groupby(["count"], observed=False, dropna=False).count().to_numpy()
        )
        # treat Nan Group
        counts = np.reshape(counts, (1, -1))[0]
        counts = np.insert(counts, 0, counts[-1])
        counts = counts[:-1]
        print("bins", self.bins)

        counts = np.cumsum(counts, axis=0)
        print("counts", counts)

        self.cdf = counts

    def selectivity(self, domain: Domain, frac=True) -> float:
        # print("domain is ", domain)
        error_rate = 1e-6
        if not domain.left:
            domain.min += error_rate * self.bin_width
            domain.left = True
        else:
            # fix issue for query with low bound equals the max bin value
            if domain.min == self.bins[-1]:
                return self.cdf[-1] - self.cdf[-2]
        if domain.right:
            domain.max += error_rate * self.bin_width
        else:
            domain.right = True
            domain.max -= 1 * self.bin_width
        # print("domain changed to ", domain)

        cnt = self._cdf(domain.max) - self._cdf(domain.min)
        if frac:
            return 1.0 * cnt / self.size
        return cnt

    def _cdf(self, x) -> float:
        idx = np.searchsorted(self.bins, x)
        # print("idx is ", idx)
        # print("x is ", x)
        if idx == 0:
            return 0.0
        if idx == len(self.bins):
            # print("meet max", self.cdf[-1])
            return float(self.cdf[-1])
        # print("-" * 50)

        return interp(self.bins[idx - 1 : idx + +1], self.cdf[idx - 1 : idx + 1], x)


class NonKeyHistogram:
    def __init__(self, top_k=5, n_bins=200, n_categorical=500) -> None:
        self.size = None  # total size, including NULL values
        self.counts = None
        self.unique_counts = None
        self.counts_no_top_k = None
        self.unique_counts_no_top_k = None
        self.counts_top_k = None
        self.unique_counts_top_k = None
        self.top_k = top_k  # the number of dominating values to maintain
        self.top_k_container = None  # a list, containing a tree of [value, counter]
        self.background_frequency = None

        self.is_categorical = False
        self.n_categorical = n_categorical
        self.n_bins = n_bins
        self.bins = None

    def fit(self, data: pd.DataFrame, headers: list) -> None:
        assert len(headers) == 1
        uniques = pd.unique(data[headers[0]])
        if len(uniques) < self.n_categorical:
            self.is_categorical = True
            self.bins = uniques
        else:
            self.bins = np.linspace(uniques[0], uniques[-1], self.n_bins)

        groups = data.groupby(pd.cut(data[headers[0]], self.bins), observed=False)
        self.counts = np.array(groups[headers[0]].count()).astype("float")

        uniques = pd.unique(data[headers[0]])
        uni = pd.DataFrame(uniques, columns=["uni"]).groupby(
            pd.cut(uniques, self.bins), observed=False
        )
        self.unique_counts = np.array(uni["uni"].count()).astype("float")

        value_counts = (
            groups.value_counts().groupby(headers[0], observed=False).head(self.top_k)
        )
        # print(type(value_counts))
        # print("value_counts\n", value_counts)
        # print("-"*80)
        top_k_container = []
        cnt = 0
        container = {}
        # cntt = 0
        for domain_value, counter in value_counts.items():
            # print(domain_value, counter)
            value = domain_value[1]
            if cnt < self.top_k:
                if counter > 1:
                    container[value] = counter
                cnt += 1
            else:
                top_k_container.append(container)
                if counter > 1:
                    container = {value: counter}
                else:
                    container = {}
                cnt = 1
        top_k_container.append(container)
        # if cntt < 20:
        #     print(domain_value[0], domain_value[1], counter)
        # cntt += 1

        self.top_k_container = top_k_container
        # print("self.top_k_container \n", self.top_k_container)
        self.counts_top_k = np.array([sum(i.values()) for i in top_k_container])
        self.unique_counts_top_k = np.array([len(i) for i in top_k_container])
        # print("self.counts_top_k \n", self.counts_top_k)
        # print("self.unique_counts_top_k \n", self.unique_counts_top_k)

        self.unique_counts_no_top_k = self.unique_counts - self.unique_counts_top_k
        self.counts_no_top_k = self.counts - self.counts_top_k
        self.background_frequency = division(
            self.counts_no_top_k * 1.0, self.unique_counts_no_top_k
        )

    def selectivity(self, domain: Domain) -> float:
        if domain.left > self.bins[-1] or domain.right < self.bins[0]:
            return 0.0

        idx_low = bisect_left(self.bins, domain.left)
        idx_high = bisect_left(self.bins, domain.right)
