import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from joins.tools import read_from_csv


class BaseHistogram:
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def join(self, hist1: "BaseHistogram") -> int:
        pass


class NormalHistogram(BaseHistogram):
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def join(self, hist1: "NormalHistogram") -> int:
        pass


class UpperBoundHistogram(BaseHistogram):
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def join(self, hist1: "UpperBoundHistogram") -> int:
        pass


class FinerHistogram(BaseHistogram):
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def join(self, hist1: "FinerHistogram") -> int:
        pass


class TableJoin(BaseHistogram):
    def __init__(self, low=None, upper=None, n=100) -> None:
        self.df = None
        self.size = None
        self.unique_size = None
        self.headers = None

    def fit(self, data: pd.DataFrame, headers: list) -> None:
        # print("headers is ", headers)
        assert len(headers) == 1
        self.headers = headers
        # print(data)
        # print(type(data))
        # self.df = data[headers]
        self.df = data
        self.size = len(self.df)

    def join(self, hist1: "TableJoin", bins) -> np.array:
        df = self.df.merge(hist1.df, left_on=self.headers, right_on=hist1.headers)
        count, bins = np.histogram(df, bins=bins)
        # print("df is \n", df)
        # print("count:\n", count)
        # print("division:\n", division)
        print("join size is ", np.sum(count))
        plt.hist(bins[:-1], bins, weights=count)
        plt.yscale("log")
        plt.show()
        return count


def plot_1d_histogram(file_path, col_header):
    data = pd.read_csv(file_path)[col_header].values
    plt.xlabel(col_header)
    plt.ylabel("number of points")
    plt.hist(data, 300, alpha=0.3, label=file_path)
    # plt.yscale("log")
    # plt.show()


if __name__ == "__main__":
    b = pd.read_csv("data/stats/badges.csv")[["UserId"]]
    c = pd.read_csv("data/stats/comments.csv")[["UserId"]]
    u = pd.read_csv("data/stats/users.csv")[["Id"]]
    # print(b.min().values[0])
    # print(c.min().values[0])
    # print(u.min().values[0])
    low = np.min([b.min().values[0], c.min().values[0], u.min().values[0]])
    high = np.max([b.max().values[0], c.max().values[0], u.max().values[0]])
    print("low ", low)
    print("high ", high)
    tj_b = TableJoin()
    tj_b.fit(b, ["UserId"])
    tj_c = TableJoin()
    tj_c.fit(c, ["UserId"])

    bins = np.linspace(low, high, 300)
    tj_b.join(tj_c, bins=bins)

    # plot_1d_histogram("data/stats/badges.csv", "UserId")
    # plot_1d_histogram("data/stats/comments.csv", "UserId")
    # plot_1d_histogram("data/stats/users.csv", "Id")
    # plt.legend()
    # plt.show()
