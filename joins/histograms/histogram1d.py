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


# def plot_1d_histogram(file_path, col_header):
#     data = pd.read_csv(file_path)[col_header].values
#     plt.xlabel(col_header)
#     plt.ylabel("number of points")
#     plt.hist(data, 300, alpha=0.3, label=file_path)
#     # plt.yscale("log")
#     # plt.show()


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

    # join-histogram
    jh_b = JoinHistogram()
    jh_b.fit(b, ["UserId"], bins)
    jh_c = JoinHistogram()
    jh_c.fit(c, ["UserId"], bins)
    jh = jh_b.join(jh_c)

    jh_error = division(jh-tj, tj)
    # print(jh_error)
    plt.hist(bins[:-1], bins, weights=jh_error)
    # plt.yscale("log")
    plt.show()

    # upperBoundHistogram
    ub_b = UpperBoundHistogram()
    ub_b.fit(b, ["UserId"], bins)
    ub_c = UpperBoundHistogram()
    ub_c.fit(c, ["UserId"], bins)
    ub = ub_b.join(ub_c)

    ub_error = division(ub-tj, tj)
    # print(ub_error)
    plt.hist(bins[:-1], bins, weights=ub_error)
    # plt.yscale("log")
    plt.show()
