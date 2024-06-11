import heapq
import pickle
import time
from collections import ChainMap
from copy import deepcopy
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joins.tools import division, read_from_csv

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def tops(s, n=3):
    return s.value_counts().head(n)


def get_dominating_items_in_histograms(top_k_container, n=1000, size=None):
    merged_dict = dict(ChainMap(*top_k_container))
    # return merged_dict
    top_n = dict(heapq.nlargest(n, merged_dict.items(), key=itemgetter(1)))
    # print("top n is ", top_n)
    return top_n


class BaseHistogram:
    def __init__(self) -> None:
        pass

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def join(self, hist1: "BaseHistogram") -> int:
        pass

    def serialize(self, name="out"):
        bytes = pickle.dumps(self, pickle.HIGHEST_PROTOCOL)
        # with open(name, "wb") as output_file:
        #     pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)
        return len(bytes)


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

    def join(self, hist1: "JoinHistogram", update_statistics=False) -> int:
        mul = np.multiply(self.counts, hist1.counts)
        maxs = np.maximum(self.unique_counts, hist1.unique_counts)
        # print("max is ", maxs)
        # counts = np.divide(mul, maxs, out=np.zeros_like(mul), where=maxs != 0)
        counts = division(mul, maxs)
        print("JoinHistogram prediction is ", np.sum(counts))
        if update_statistics:
            self.counts = counts
            self.unique_counts = np.minimum(self.unique_counts, hist1.unique_counts)
            return self
        return counts


class UpperBoundHistogram(BaseHistogram):
    def __init__(self) -> None:
        self.counts = None
        self.mfv_counts = None

    def fit(self, data: pd.DataFrame, headers: list, bins) -> None:
        groups = data.groupby(pd.cut(data[headers[0]], bins), observed=False)
        self.counts = np.array(groups[headers[0]].count()).astype("float")

        value_counts = groups.value_counts().groupby(headers[0], observed=False).head(1)
        # print("value_counts\n", value_counts)

        mfv_counts = np.array(value_counts)
        # print("mfv_counts\n", mfv_counts)
        self.mfv_counts = mfv_counts.astype("float")

    def join(self, hist1: "UpperBoundHistogram", update_statistics=False) -> int:
        res = np.minimum(
            division(self.counts, self.mfv_counts),
            division(hist1.counts, hist1.mfv_counts),
        )
        res = np.multiply(res, self.mfv_counts)
        res = np.multiply(res, hist1.mfv_counts)
        print("UpperBoundHistogram prediction is ", np.sum(res))
        if update_statistics:
            hist = deepcopy(self)

        return res


class UpperBoundHistogramTopK(BaseHistogram):
    def __init__(self, top_k=5) -> None:
        self.counts = None
        self.unique_counts = None
        self.counts_no_top_k = None
        self.unique_counts_no_top_k = None
        self.counts_top_k = None
        self.unique_counts_top_k = None
        self.top_k = top_k  # the number of dominating values to maintain
        # a list, containing a tree of [value, counter]
        self.top_k_container = None
        self.background_frequency = None

    def fit(self, data: pd.DataFrame, headers: list, bins) -> None:
        groups = data.groupby(
            pd.cut(data[headers[0]], bins, include_lowest=True), observed=False
        )
        self.counts = np.array(groups[headers[0]].count()).astype("float")

        uniques = pd.unique(data[headers[0]])
        uni = pd.DataFrame(uniques, columns=["uni"]).groupby(
            pd.cut(uniques, bins, include_lowest=True), observed=False
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
        # print("background_frequency", self.background_frequency)

        # mfv_counts = np.array(value_counts)
        # # print("mfv_counts\n", mfv_counts)
        # self.mfv_counts = mfv_counts.astype("float")

    def join(
        self, hist1: "UpperBoundHistogramTopK", id_filtered=None
    ) -> "UpperBoundHistogramTopK":
        # start = time.time()
        # not top k
        mul = np.multiply(self.counts_no_top_k, hist1.counts_no_top_k)
        maxs = np.maximum(self.unique_counts_no_top_k, hist1.unique_counts_no_top_k)
        counts_no_top_k = division(mul, maxs)

        # top k
        counts_top_k = []
        top_k_container = []
        for aa, bb, fa, fb in zip(
            self.top_k_container,
            hist1.top_k_container,
            self.background_frequency,
            hist1.background_frequency,
        ):
            set_a = set(aa)
            set_b = set(bb)
            id_filtered = set(id_filtered) if id_filtered else set()
            # if id_filtered:
            #     set_a = set_a.intersection(set(id_filtered))
            #     set_b = set_b.intersection(set(id_filtered))
            # cnt = 0
            container = {}
            # print("-"*80)
            # if strategy == "keep":
            common_ids = (
                set_a.intersection(set_b) - id_filtered
                if id_filtered
                else set_a.intersection(set_b)
            )
            for k in common_ids:
                container[k] = aa[k] * bb[k]
            # print("keys are ", container.keys())
            # TODO: proper handling here, need to store more data for skewed data!!
            # if strategy == "keep":
            common_a_not_b = (
                set_a - set_a.intersection(set_b) - id_filtered
                if id_filtered
                else set_a - set_a.intersection(set_b)
            )
            for k in common_a_not_b:
                container[k] = aa[k] * fb
            common_b_not_a = (
                set_b - set_a.intersection(set_b) - id_filtered
                if id_filtered
                else set_b - set_a.intersection(set_b)
            )
            for k in common_b_not_a:
                container[k] = bb[k] * fa
                # cnt += aa[k] * bb[k]
                # counts_top_k.append(cnt)
            top_k_container.append(container)
        # counts_top_k = np.array(counts_top_k)
        counts_top_k = np.array([sum(i.values()) for i in top_k_container])
        unique_counts_top_k = np.array([len(i) for i in top_k_container])

        counts = np.add(counts_top_k, counts_no_top_k)

        # if update_statistics:
        hist = deepcopy(self)
        hist.counts_no_top_k = counts_no_top_k
        hist.counts_top_k = counts_top_k
        hist.counts = counts  # np.add(counts_no_top_k, counts_top_k)
        hist.top_k_container = top_k_container
        hist.unique_counts_top_k = unique_counts_top_k
        hist.unique_counts = np.minimum(self.unique_counts, hist1.unique_counts)
        hist.unique_counts_no_top_k = self.unique_counts - self.unique_counts_top_k
        hist.background_frequency = division(
            hist.counts_no_top_k * 1.0, hist.unique_counts_no_top_k
        )
        # end = time.time()
        # print(
        #     "UpperBoundHistogramTopK prediction is ",
        #     np.sum(counts),
        #     "with time cost ",
        #     end - start,
        #     " seconds, and size ",
        #     self.serialize(),
        #     " bytes.",
        # )
        # if update_statistics:
        #     return hist

        # return counts
        return hist


# class UpperBoundHistogramTopK2D(BaseHistogram):
#     def __init__(self, top_k=5) -> None:
#         self.counts = None
#         self.unique_counts = None
#         self.counts_no_top_k = None
#         self.unique_counts_no_top_k = None
#         self.counts_top_k = None
#         self.unique_counts_top_k = None
#         self.top_k = top_k  # the number of dominating values to maintain
#         self.top_k_container = None  # a list, containing a tree of [value, counter]
#         self.background_frequency = None

#     def fit(self, data: pd.DataFrame, headers: list, grid_x, grid_y) -> None:
#         assert len(headers) == 2
#         print(len(data))
#         data = data.astype(float)
#         data = data.dropna()
#         print(len(data))
#         df = data.assign(
#             x_cut=pd.cut(data[headers[0]], grid_x, include_lowest=True),
#             y_cut=pd.cut(data[headers[1]], grid_y, include_lowest=True),
#         )
#         print("df\n", df)
#         groups = df.groupby(["x_cut", "y_cut"], observed=False)
#         # print(groups[headers].count())
#         # print(groups.size().unstack())
#         tmp = groups.size().unstack().to_numpy()
#         print(tmp)

#         self.counts = groups.size().unstack().to_numpy().astype("float")

#         # groups = data.groupby(pd.cut(data[headers], bins), observed=False)
#         # self.counts = np.array(groups[headers[0]].count()).astype("float")

#         uniques = df.groupby(headers).size().reset_index().rename(columns={0: "count"})
#         uni = uniques.assign(
#             x_cut=pd.cut(uniques[headers[0]], grid_x, include_lowest=True),
#             y_cut=pd.cut(uniques[headers[1]], grid_y, include_lowest=True),
#         )
#         uni_groups = uni.groupby(
#             ["x_cut", "y_cut"], observed=False
#         )  # .size().unstack().to_numpy()
#         # print("uniques\n", uniques.head(30))
#         # print("uni_groups\n", uni_groups)
#         # uni = uniques.assign(
#         #     x_cut=pd.cut(data[headers[0]], grid_x),
#         #     y_cut=pd.cut(data[headers[1]], grid_y),
#         # )
#         self.unique_counts = uni_groups.size().unstack().to_numpy().astype("float")
#         print("self.unique_counts", self.unique_counts)
#         # print(self.unique_counts)
#         # TODO unique matix is abnormal, compared with counts matrix

#         print("groups", groups)
#         # print("value counts", groups.value_counts())
#         print("size is ", len(groups))
#         # value_counts = groups.head(5)
#         value_counts = groups[].apply(tops, n=self.top_k)
#         print("value_counts\n", value_counts)
#         # cc=0
#         # for k, v in groups:
#         #     print(k, v)
#         exit()
#         # value_counts = (
#         #     groups.value_counts().groupby(headers, observed=False).head(self.top_k)
#         # )
#         # print(type(value_counts))
#         # print("value_counts\n", value_counts)
#         # print("-"*80)
#         top_k_container = []
#         cnt = 0
#         container = {}
#         # cntt = 0
#         for domain_value, counter in value_counts.items():
#             # print(domain_value, counter)
#             value = domain_value[1]
#             if cnt < self.top_k:
#                 if counter > 1:
#                     container[value] = counter
#                 cnt += 1
#             else:
#                 top_k_container.append(container)
#                 if counter > 1:
#                     container = {value: counter}
#                 else:
#                     container = {}
#                 cnt = 1
#         top_k_container.append(container)
#         # if cntt < 20:
#         #     print(domain_value[0], domain_value[1], counter)
#         # cntt += 1

#         self.top_k_container = top_k_container
#         # print("self.top_k_container \n", self.top_k_container)
#         self.counts_top_k = np.array([sum(i.values()) for i in top_k_container])
#         self.unique_counts_top_k = np.array([len(i) for i in top_k_container])
#         # print("self.counts_top_k \n", self.counts_top_k)
#         # print("self.unique_counts_top_k \n", self.unique_counts_top_k)

#         self.unique_counts_no_top_k = self.unique_counts - self.unique_counts_top_k
#         self.counts_no_top_k = self.counts - self.counts_top_k
#         self.background_frequency = division(
#             self.counts_no_top_k * 1.0, self.unique_counts_no_top_k
#         )
#         # print("background_frequency", self.background_frequency)

#         # mfv_counts = np.array(value_counts)
#         # # print("mfv_counts\n", mfv_counts)
#         # self.mfv_counts = mfv_counts.astype("float")

#     def join(self, hist1: "UpperBoundHistogramTopK", update_statistics=False) -> int:
#         start = time.time()
#         # not top k
#         mul = np.multiply(self.counts_no_top_k, hist1.counts_no_top_k)
#         maxs = np.maximum(self.unique_counts_no_top_k, hist1.unique_counts_no_top_k)
#         counts_no_top_k = division(mul, maxs)

#         # top k
#         counts_top_k = []
#         top_k_container = []
#         for aa, bb, fa, fb in zip(
#             self.top_k_container,
#             hist1.top_k_container,
#             self.background_frequency,
#             hist1.background_frequency,
#         ):
#             set_a = set(aa)
#             set_b = set(bb)
#             # cnt = 0
#             container = {}
#             for k in set_a.intersection(set_b):
#                 container[k] = aa[k] * bb[k]
#             for k in set_a - set_a.intersection(set_b):
#                 container[k] = aa[k] * fb
#             for k in set_b - set_a.intersection(set_b):
#                 container[k] = bb[k] * fa
#                 # cnt += aa[k] * bb[k]
#             # counts_top_k.append(cnt)
#             top_k_container.append(container)
#         # counts_top_k = np.array(counts_top_k)
#         counts_top_k = np.array([sum(i.values()) for i in top_k_container])
#         unique_counts_top_k = np.array([len(i) for i in top_k_container])

#         counts = np.add(counts_top_k, counts_no_top_k)

#         if update_statistics:
#             hist = deepcopy(self)
#             hist.counts_no_top_k = counts_no_top_k
#             hist.counts_top_k = counts_top_k
#             hist.counts = np.add(counts_no_top_k, counts_top_k)
#             hist.top_k_container = top_k_container
#             hist.unique_counts_top_k = unique_counts_top_k
#             hist.unique_counts = np.minimum(self.unique_counts, hist1.unique_counts)
#             hist.unique_counts_no_top_k = self.unique_counts - self.unique_counts_top_k
#             hist.background_frequency = division(
#                 hist.counts_no_top_k * 1.0, hist.unique_counts_no_top_k
#             )
#         end = time.time()
#         print(
#             "UpperBoundHistogramTopK prediction is ",
#             np.sum(counts),
#             "with time cost ",
#             end - start,
#             " seconds, and size ",
#             self.serialize(),
#             " bytes.",
#         )
#         if update_statistics:
#             return hist

#         return counts


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
        df = self.df.merge(hist1.df, left_on=self.headers, right_on=hist1.headers)
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
    ph = pd.read_csv("data/stats/postHistory.csv")[["UserId"]]
    p = pd.read_csv("data/stats/posts.csv")[["OwnerUserId"]]

    low = np.min(
        [
            b.min().values[0],
            c.min().values[0],
            u.min().values[0],
            ph.min().values[0],
            p.min().values[0],
        ]
    )
    high = np.max(
        [
            b.max().values[0],
            c.max().values[0],
            u.max().values[0],
            ph.max().values[0],
            p.max().values[0],
        ]
    )
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

    jh_error = division(jh, tj)
    # print(jh_error)
    # plt.hist(bins[:-1], bins, weights=jh_error, label="Join-Histogram")
    # plt.yscale("log")
    # plt.show()

    # upperBoundHistogram
    ub_b = UpperBoundHistogram()
    ub_b.fit(b, ["UserId"], bins)
    ub_c = UpperBoundHistogram()
    ub_c.fit(c, ["UserId"], bins)
    ub = ub_b.join(ub_c)
    print("estimation from upper bound ", np.sum(ub))

    plt.figure(dpi=300)
    ub_error = division(ub, tj)
    # print(ub_error)
    plt.hist(bins[:-1], bins, weights=ub_error, label="Upper Bound", color=colors[2])
    # plt.yscale("log")
    # plt.show()

    # upperBoundHistogramTopK
    ubtk_b = UpperBoundHistogramTopK(10)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(10)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_b.join(ubtk_c)

    print("estimation from top K ", np.sum(ubtk))

    ubtk_error = division(ubtk, tj)
    # print(ub_error)
    plt.hist(bins[:-1], bins, weights=ubtk_error, label="DHist", color=colors[0])
    # plt.yscale("log")

    plt.hist(bins[:-1], bins, weights=jh_error, label="Join-Histogram", color=colors[1])
    plt.legend()
    plt.xlabel("Join Key: UserId")
    plt.ylabel("Accuracy: predction/truth")

    plt.show()

    # ubtk_b = UpperBoundHistogramTopK(3)
    # ubtk_b.fit(b, ["UserId"], bins)
    # ubtk_c = UpperBoundHistogramTopK(3)
    # ubtk_c.fit(c, ["UserId"], bins)
    # ubtk = ubtk_b.join(ubtk_c)

    # ubtk_b = UpperBoundHistogramTopK(5)
    # ubtk_b.fit(b, ["UserId"], bins)
    # ubtk_c = UpperBoundHistogramTopK(5)
    # ubtk_c.fit(c, ["UserId"], bins)
    # ubtk = ubtk_b.join(ubtk_c)

    # ubtk_b = UpperBoundHistogramTopK(10)
    # ubtk_b.fit(b, ["UserId"], bins)
    # ubtk_c = UpperBoundHistogramTopK(10)
    # ubtk_c.fit(c, ["UserId"], bins)
    # ubtk = ubtk_b.join(ubtk_c)

    # ubtk_b = UpperBoundHistogramTopK(20)
    # ubtk_b.fit(b, ["UserId"], bins)
    # ubtk_c = UpperBoundHistogramTopK(20)
    # ubtk_c.fit(c, ["UserId"], bins)
    # ubtk = ubtk_b.join(ubtk_c)

    # ubtk_b = UpperBoundHistogramTopK(100)
    # ubtk_b.fit(b, ["UserId"], bins)
    # ubtk_c = UpperBoundHistogramTopK(100)
    # ubtk_c.fit(c, ["UserId"], bins)
    # ubtk = ubtk_b.join(ubtk_c)

    # ubtk_error = ubtk - tj  # division(ubtk-tj, tj)
    # # print(ubtk_error)
    # plt.hist(bins[:-1], bins, weights=ubtk_error)
    # # plt.yscale("log")
    # plt.show()

    # ubtk_ph = UpperBoundHistogramTopK(5)
    # ubtk_ph.fit(ph, ["UserId"], bins)
    # ubtk_c = UpperBoundHistogramTopK(5)
    # ubtk_c.fit(c, ["UserId"], bins)
    # ubtk = ubtk_ph.join(ubtk_c)

    # SELECT COUNT(*)  FROM badges as b,  posts as p,  users as u  WHERE u.Id = p.OwnerUserId   AND u.Id = b.UserId
    # 3728360
    # SELECT COUNT(*)  FROM badges as b,  posts as p,  comments as c WHERE c.UserId = p.OwnerUserId   AND c.UserId = b.UserId
    # 15131840763
    # SELECT COUNT(*)  FROM badges as b,  posts as p,  users as u ,  comments as c WHERE u.Id = p.OwnerUserId   AND u.Id = b.UserId AND c.UserId = b.UserId
    #
    ubtk_u = UpperBoundHistogramTopK(10)
    ubtk_u.fit(u, ["Id"], bins)
    ubtk_b = UpperBoundHistogramTopK(10)
    ubtk_b.fit(b, ["UserId"], bins)
    ubtk_p = UpperBoundHistogramTopK(10)
    ubtk_p.fit(p, ["OwnerUserId"], bins)
    ubtk_c = UpperBoundHistogramTopK(10)
    ubtk_c.fit(c, ["UserId"], bins)
    ubtk = ubtk_p.join(ubtk_b, update_statistics=True).join(ubtk_c)

    # SELECT COUNT(*)  FROM badges as b,  users as u  WHERE  u.Id = b.UserId
    # 79851

    # SELECT COUNT(*)  FROM badges as b,  posts as p WHERE p.OwnerUserId = b.UserId
    # 3728360
