import random
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


def interp_dominating_item(dic: dict, domain: Domain):
    filtered = [v for k, v in dic.items() if domain.contain(k)]
    print("kv", filtered)
    return np.sum(filtered)


class NonKeyCumulativeHistogram:
    def __init__(self, n_bins=100, n_categorical=50, n_top_k=30) -> None:
        # assert n_bins >= n_top_k
        self.n_top_k = n_top_k
        self.n_bins = n_bins
        self.n_categorical = n_categorical
        self.bins = None
        self.is_categorical = False
        self.cdf = None
        self.size = None
        self.bin_width = 1.0  # for categorical attributes, the default width is 1.0, modify it  if necessary

    def fit(self, data: pd.DataFrame, headers: list) -> None:
        assert len(headers) == 1
        self.size = data.shape[0]
        uniques = np.sort(pd.unique(data[headers[0]]))
        if len(uniques) < self.n_categorical:
            self.is_categorical = True
            self.bins = uniques
        else:
            self.bins = np.linspace(uniques[0], uniques[-1], self.n_bins)
            self.bin_width = self.bins[1] - self.bins[0]
            # tops = data.value_counts().head(self.n_top_k).index.to_list()
            # tops = [i[0] for i in tops]
            # print("value counts", self.n_top_k)
            # print(tops)
            # extras = random.sample(
            #     list(set(uniques).difference(set(tops))), self.n_bins - self.n_top_k
            # )
            # print(extras)
            # keys = list(set(tops + extras + [uniques[0], uniques[-1]]))
            # print(keys)
            # self.bins = np.array(sorted(keys))
        # print("data", data)
        data["count"] = pd.cut(data[headers[0]], bins=self.bins, labels=self.bins[:-1])
        # print(data)
        counts = (
            data.groupby(["count"], observed=False, dropna=False).count().to_numpy()
        )
        # print(data.groupby(["count"], observed=False, dropna=False).count())
        # treat Nan Group
        counts = np.reshape(counts, (1, -1))[0]
        counts = np.insert(counts, 0, counts[-1])
        counts = counts[:-1]
        # print("bins", self.bins)
        # print("counts", counts)

        counts = np.cumsum(counts, axis=0)
        # print("counts", counts)

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
                return (
                    (self.cdf[-1] - self.cdf[-2]) * 1.0 / (self.bins[1] - self.bins[0])
                )
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


class NonKeyTopKHistogram:
    def __init__(self, n_top_k=30, n_bins=100, n_categorical=50) -> None:
        self.n_top_k = n_top_k
        self.n_bins = n_bins
        self.n_categorical = n_categorical
        self.bins = None
        self.is_categorical = False
        self.cdf = None
        self.size = None
        self.bin_width = 1.0  # for categorical attributes, the default width is 1.0, modify it  if necessary
        self.bins = None

        self.counts = None
        self.unique_counts = None
        self.counts_no_top_k = None
        self.unique_counts_no_top_k = None
        self.counts_top_k = None
        self.unique_counts_top_k = None
        self.top_k_container = None  # a list, containing trees of [value, counter]
        self.background_frequency = None
        self.min = None
        self.max = None

    def fit(self, data: pd.DataFrame, headers: list) -> None:
        assert len(headers) == 1
        self.size = data.shape[0]
        uniques = np.sort(pd.unique(data[headers[0]]))
        self.min = uniques[0]
        self.max = uniques[-1]
        # print("uniques,", uniques)
        if len(uniques) < self.n_categorical:
            self.is_categorical = True
            self.bins = uniques
        else:
            self.bins = np.linspace(uniques[0], uniques[-1], self.n_bins)
            self.bin_width = self.bins[1] - self.bins[0]
            # tops = data.value_counts().head(self.n_top_k).index.to_list()
            # tops = [i[0] for i in tops]
            # print("value counts", self.n_top_k)
            # print(tops)
            # extras = random.sample(
            #     list(set(uniques).difference(set(tops))), self.n_bins - self.n_top_k
            # )
            # print(extras)
            # keys = tops + extras
            # print(keys)
            # self.bins = np.array(sorted(keys))
        # print("self.bins", self.bins)
        groups = data.groupby(
            pd.cut(data[headers[0]], self.bins, include_lowest=True), observed=False
        )
        self.counts = np.array(groups[headers[0]].count()).astype("float")
        # print("countsssss", groups[headers[0]].count())

        uniques = pd.unique(data[headers[0]])
        uni = pd.DataFrame(uniques, columns=["uni"]).groupby(
            pd.cut(uniques, self.bins, include_lowest=True), observed=False
        )
        self.unique_counts = np.array(uni["uni"].count()).astype("float")

        value_counts = (
            groups.value_counts().groupby(headers[0], observed=False).head(self.n_top_k)
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
            if cnt < self.n_top_k:
                if counter >= 1:
                    container[value] = counter
                cnt += 1
            else:
                top_k_container.append(container)
                if counter >= 1:
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
        print("bins", self.bins)
        print("counts", self.counts)
        print("unique_counts", self.unique_counts)
        print("top_k_container", self.top_k_container)
        print("backgroud", self.background_frequency)
        print("notopk ", self.counts_no_top_k)

    def selectivity(self, domain: Domain, frac=True) -> float:
        if np.isinf(domain.max):
            domain.max = self.max
            domain.right = True

        if np.isneginf(domain.min):
            domain.min = self.min
            domain.left = True

        if domain.max < self.bins[0]:
            cnt = 0.0
        elif domain.min > self.bins[-1]:
            cnt = 0.0
        elif domain.max == self.bins[0]:
            if domain.right:
                if domain.max in self.top_k_container[0]:
                    cnt = self.top_k_container[0][domain.max]
                else:
                    cnt = self.background_frequency[0]
            else:
                cnt = 0.0
        elif domain.min == self.bins[-1]:
            if domain.left:
                if domain.min in self.top_k_container[-1]:
                    cnt = self.top_k_container[-1][domain.min]
                else:
                    cnt = self.background_frequency[-1]
            else:
                cnt = 0.0
        else:
            # normal cases
            # print("domain is ", domain)
            # error_rate = 1e-6
            # if not domain.left:
            #     domain.min += error_rate * self.bin_width
            #     domain.left = True
            # else:
            #     # fix issue for query with low bound equals the max bin value
            #     if domain.min == self.bins[-1]:
            #         return (
            #             (self.cdf[-1] - self.cdf[-2]) * 1.0 / (self.bins[1] - self.bins[0])
            #         )
            # if domain.right:
            #     domain.max += error_rate * self.bin_width
            # else:
            #     domain.right = True
            #     domain.max -= 1 * self.bin_width
            # print("domain changed to ", domain)

            idxs = np.searchsorted(self.bins, [domain.min, domain.max])
            print("idxs is ", idxs)
            idx_left = max(0, idxs[0] - 1)
            idx_right = min(idxs[1] - 1, len(self.counts) - 1)
            if idx_right - idx_left > 1:
                cnt = np.sum(self.counts[idxs[0] : idxs[1] - 1])
            else:
                cnt = 0.0
            print("middle is ", self.counts[idxs[0] : idxs[1] - 1])
            print("sum asdfasdf ", cnt)
            print("left bin ", idx_left, self.counts[idx_left])
            print("right bin ", idx_right, self.counts[idx_right])

            # interpret left incomplete bin
            # dominating term
            # if domain.min > self.bins[0]:
            cnt += interp_dominating_item(self.top_k_container[idx_left], domain)
            print(
                "dominating plus:",
                interp_dominating_item(self.top_k_container[idx_left], domain),
            )
            if idx_right > idx_left:  # and not np.isinf(domain.max):
                cnt += interp_dominating_item(self.top_k_container[idx_right], domain)
                print(
                    "dominating plus:",
                    interp_dominating_item(self.top_k_container[idx_right], domain),
                )
            print("dominating:", cnt)
            # uniform assumption term
            if idx_left == idx_right:
                cnt += (
                    1.0
                    * self.counts_no_top_k[idx_right]
                    / self.bin_width
                    * (domain.max - domain.min)
                )
            else:
                # add right part
                print("domain is ", domain)
                # if np.isinf(domain.max):
                #     print("is inf ", self.counts_no_top_k[-1])
                #     cnt -= self.counts_no_top_k[-1]
                #     print("cnt changed to ", cnt)
                # else:
                cnt += (
                    1.0
                    * self.counts_no_top_k[idx_right]
                    / self.bin_width
                    * (domain.max - self.bins[idx_right])
                )
                # add left part
                cnt += (
                    1.0
                    * self.counts_no_top_k[idx_left]
                    / self.bin_width
                    * (self.bins[idx_left + 1] - domain.min)
                )

            # # if idxs[0] == idxs[1]:
            # #     cnt = self.counts[idxs[0]]
            # # else:
            # #     cnt = np.sum(self.counts[idxs[0] : idxs[1]])

            # # print("temp sum is ", cnt)
            # # if cnt == 0.0:
            # #     return 0.0

            # if domain.min == self.bins[idxs[0]]:
            #     if not domain.left:
            #         ids = idxs[0]
            #         if ids >= len(self.background_frequency):
            #             ids = len(self.background_frequency) - 1
            #         val = self.background_frequency[ids]
            #         if domain.min in self.top_k_container[ids]:
            #             val = self.top_k_container[ids][domain.min]
            #         print("left val is ", val)
            #         cnt -= val
            # if idxs[1] < len(self.bins):
            #     if domain.max == self.bins[idxs[1]]:
            #         # print("hehre")
            #         if not domain.right:
            #             ids = idxs[1]
            #             # if ids >= len(self.background_frequency):
            #             #     ids = len(self.background_frequency) - 1
            #             val = self.background_frequency[ids]
            #             # print(self.top_k_container[idxs[1]])
            #             # print(self.top_k_container[idxs[1] + 1])
            #             if domain.max in self.top_k_container[idxs[1] - 1]:
            #                 val = self.top_k_container[idxs[1] - 1][domain.max]
            #             # print("val is ", val)
            #             print("right val is ", val)
            #             cnt -= val
            # else:
            #     pass  # do not need to do anything for such situations.

        # cnt = self._cdf(domain.max) - self._cdf(domain.min)
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
            return float(np.sum(self.counts))
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
