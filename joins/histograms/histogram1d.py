import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from joins.tools import read_from_csv


def division(x: np.array, y: np.array):
    return np.divide(x, y, out=np.zeros_like(x), where=y != 0)


class BaseHistogram:
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def join(self, hist1: "BaseHistogram") -> int:
        pass


class JoinHistogram(BaseHistogram):
    def __init__(self) -> None:
        self.headers = None
        self.counts = None
        self.unique_counts = None

    def fit(self, data: pd.DataFrame, headers: list, bins) -> None:
        # print(data)
        groups = data.groupby(pd.cut(data[headers[0]], bins), observed=False)
        self.counts = np.array(groups[headers[0]].count()).astype("float")

        uniques = pd.unique(data[headers[0]])
        # print("uniques\n", uniques)
        # uni = pd.cut(uniques, bins=bins)  # , labels=self.grid_x[:-1]
        uni = pd.DataFrame(uniques, columns=["uni"]).groupby(
            pd.cut(uniques, bins), observed=False
        )
        self.unique_counts = np.array(uni["uni"].count()).astype("float")
        # print("self.unique_counts\n", self.unique_counts)

    def join(self, hist1: "JoinHistogram") -> int:
        mul = np.multiply(self.counts, hist1.counts)
        maxs = np.maximum(self.unique_counts,
                          hist1.unique_counts)
        # print("max is ", maxs)
        # counts = np.divide(mul, maxs, out=np.zeros_like(mul), where=maxs != 0)
        counts = division(mul, maxs)
        print("JoinHistogram prediction is ", np.sum(counts))
        return counts


class UpperBoundHistogram(BaseHistogram):
    def __init__(self) -> None:
        self.counts = None
        self.mfv_counts = None

    def fit(self, data: pd.DataFrame, headers: list, bins) -> None:
        groups = data.groupby(pd.cut(data[headers[0]], bins), observed=False)
        self.counts = np.array(groups[headers[0]].count()).astype("float")

        value_counts = groups.value_counts().groupby(
            headers[0], observed=False).head(1)
        # print("value_counts\n", value_counts)

        mfv_counts = np.array(value_counts)
        # print("mfv_counts\n", mfv_counts)
        self.mfv_counts = mfv_counts.astype("float")

    def join(self, hist1: "UpperBoundHistogram") -> int:
        res = np.minimum(division(self.counts, self.mfv_counts),
                         division(hist1.counts, hist1.mfv_counts))
        res = np.multiply(res, self.mfv_counts)
        res = np.multiply(res, hist1.mfv_counts)
        print("UpperBoundHistogram prediction is ", np.sum(res))
        return res


class UpperBoundHistogramTopK(BaseHistogram):
    def __init__(self, top_k=5) -> None:
        self.counts = None
        self.unique_counts = None
        self.counts_no_top_k = None
        self.unique_counts_no_top_k = None
        self.counts_top_k = None
        self.unique_counts_top_k = None
        # self.mfv_counts = None
        self.top_k = top_k
        self.top_k_container = None

    def fit(self, data: pd.DataFrame, headers: list, bins) -> None:
        groups = data.groupby(pd.cut(data[headers[0]], bins), observed=False)
        self.counts = np.array(groups[headers[0]].count()).astype("float")

        uniques = pd.unique(data[headers[0]])
        uni = pd.DataFrame(uniques, columns=["uni"]).groupby(
            pd.cut(uniques, bins), observed=False
        )
        self.unique_counts = np.array(uni["uni"].count()).astype("float")

        value_counts = groups.value_counts().groupby(
            headers[0], observed=False).head(self.top_k)
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
                if counter > 0:
                    container[value] = counter
                cnt += 1
            else:
                top_k_container.append(container)
                if counter > 0:
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
        self.counts_top_k = np.array([sum(i.values())
                                     for i in top_k_container])
        self.unique_counts_top_k = np.array([len(i) for i in top_k_container])
        # print("self.counts_top_k \n", self.counts_top_k)
        # print("self.unique_counts_top_k \n", self.unique_counts_top_k)

        self.unique_counts_no_top_k = self.unique_counts-self.unique_counts_top_k
        self.counts_no_top_k = self.counts-self.counts_top_k

        # mfv_counts = np.array(value_counts)
        # # print("mfv_counts\n", mfv_counts)
        # self.mfv_counts = mfv_counts.astype("float")

    def join(self, hist1: "UpperBoundHistogramTopK") -> int:
        # not top k
        mul = np.multiply(self.counts_no_top_k, hist1.counts_no_top_k)
        maxs = np.maximum(self.unique_counts_no_top_k,
                          hist1.unique_counts_no_top_k)
        counts_no_top_k = division(mul, maxs)

        # top k
        counts_top_k = []
        for aa, bb in zip(self.top_k_container, hist1.top_k_container):
            set_a = set(aa)
            set_b = set(bb)
            cnt = 0
            for k in set_a.intersection(set_b):
                cnt += aa[k]*bb[k]
            counts_top_k.append(cnt)
        counts_top_k = np.array(counts_top_k)

        counts = np.add(counts_top_k, counts_no_top_k)
        print("UpperBoundHistogramTopK prediction is ", np.sum(counts))
        return counts


class FinerHistogram(BaseHistogram):
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def join(self, hist1: "FinerHistogram") -> int:
        pass


class TableJoin(BaseHistogram):
    def __init__(self) -> None:
        self.df = None
        self.size = None
        self.unique_size = None
        self.headers = None

    def fit(self, data: pd.DataFrame, headers: list) -> None:
        assert len(headers) == 1
        self.headers = headers
        # print(data)
        # print(type(data))
        # self.df = data[headers]
        self.df = data
        self.size = len(self.df)

    def join(self, hist1: "TableJoin", bins) -> np.array:
        df = self.df.merge(hist1.df, left_on=self.headers,
                           right_on=hist1.headers)
        count, bins = np.histogram(df, bins=bins)
        # # print("df is \n", df)
        # # print("count:\n", count)
        # # print("division:\n", division)
        print("join size is ", np.sum(count))
        # plt.hist(bins[:-1], bins, weights=count)
        # plt.yscale("log")
        # plt.show()
        return count


if __name__ == "__main__":
    b = pd.read_csv("data/stats/badges.csv")[["UserId"]]
    c = pd.read_csv("data/stats/comments.csv")[["UserId"]]
    u = pd.read_csv("data/stats/users.csv")[["Id"]]

    low = np.min([b.min().values[0], c.min().values[0], u.min().values[0]])
    high = np.max([b.max().values[0], c.max().values[0], u.max().values[0]])
    print("low ", low)
    print("high ", high)
    bins = np.linspace(low, high, 300)

    # truth
    tj_b = TableJoin()
    tj_b.fit(b, ["UserId"])
    tj_c = TableJoin()
    tj_c.fit(c, ["UserId"])
    tj = tj_b.join(tj_c, bins=bins)

    # # join-histogram
    # jh_b = JoinHistogram()
    # jh_b.fit(b, ["UserId"], bins)
    # jh_c = JoinHistogram()
    # jh_c.fit(c, ["UserId"], bins)
    # jh = jh_b.join(jh_c)

    # jh_error = division(jh-tj, tj)
    # # print(jh_error)
    # plt.hist(bins[:-1], bins, weights=jh_error)
    # # plt.yscale("log")
    # plt.show()

    # # upperBoundHistogram
    # ub_b = UpperBoundHistogram()
    # ub_b.fit(b, ["UserId"], bins)
    # ub_c = UpperBoundHistogram()
    # ub_c.fit(c, ["UserId"], bins)
    # ub = ub_b.join(ub_c)

    # ub_error = division(ub-tj, tj)
    # # print(ub_error)
    # plt.hist(bins[:-1], bins, weights=ub_error)
    # # plt.yscale("log")
    # plt.show()

    # upperBoundHistogramTopK
    ubtk_b = UpperBoundHistogramTopK(1)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(1)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_b.join(ubtk_c)

    ubtk_b = UpperBoundHistogramTopK(3)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(3)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_b.join(ubtk_c)

    ubtk_b = UpperBoundHistogramTopK(5)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(5)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_b.join(ubtk_c)

    ubtk_b = UpperBoundHistogramTopK(10)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(10)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_b.join(ubtk_c)

    ubtk_b = UpperBoundHistogramTopK(20)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(20)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_b.join(ubtk_c)

    ubtk_b = UpperBoundHistogramTopK(100)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(100)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_b.join(ubtk_c)

    ubtk_error = ubtk-tj  # division(ubtk-tj, tj)
    # print(ubtk_error)
    plt.hist(bins[:-1], bins, weights=ubtk_error)
    # plt.yscale("log")
    plt.show()
