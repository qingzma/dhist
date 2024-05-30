from bisect import bisect_left

import numpy as np
import pandas as pd

from joins.domain import Domain
from joins.tools import division


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
