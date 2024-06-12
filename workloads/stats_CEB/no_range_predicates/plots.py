# python run.py --evaluate --model models/model_stats_200_20.pkl --query workloads/stats_CEB/no_range_predicates/6.sql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joins.tools import read_from_csv, read_from_csv_to_series

plt.figure(dpi=300)


def read_data(first: str, second: str):
    truths = read_from_csv_to_series(
        "workloads/stats_CEB/no_range_predicates/" + first + ".csv", "truth"
    )
    est = read_from_csv_to_series(
        "workloads/stats_CEB/no_range_predicates/" + second + ".csv", "card"
    )

    truth1 = truths[~pd.isnull(truths)]
    est1 = est[~pd.isnull(truths)]
    # print(truths)
    # print(est)
    # idx1 = np.array([not np.isnan(truths)][0])
    # idx2 = np.array([not np.isnan(truths)][0])
    # idx = np.where(idx1 & idx2)

    # print("idx is ", idx)
    # card = card[idx]
    # truths = truths[idx]
    return truth1.values, est1.values


def read_times(first: str, header: str):
    truths = read_from_csv_to_series(
        "workloads/stats_CEB/no_range_predicates/" + first + ".csv", header
    )

    truth1 = truths[~pd.isnull(truths)]
    return truth1.values


def plot_accuracy():
    truths1, card1 = read_data("1.truth", "1.card")
    truths2, card2 = read_data("2.truth", "2.card")
    truths3, card3 = read_data("3.truth", "3.card")
    truths4, card4 = read_data("4.truth", "4.card")
    truths5, card5 = read_data("5.truth", "5.card")

    # postgres = read_from_csv("results/stats/multiple_tables/postgres.csv", "postgres")
    # bayescard = read_from_csv("results/stats/multiple_tables/bayescard.csv", "bayescard")
    # wjsample = read_from_csv("results/stats/multiple_tables/wjsample.csv", "wjsample")
    # deepdb = read_from_csv("results/stats/multiple_tables/deepdb.csv", "deepdb")
    x = range(1, 6)
    ac1 = np.average(card1 / truths1)
    ac2 = np.average(card2 / truths2)
    ac3 = np.average(card3 / truths3)
    ac4 = np.average(card4 / truths4)
    ac5 = np.average(card5 / truths5)
    plt.axhline(y=1, color="r", linestyle="--")
    print(ac1, ac2, ac3, ac4, ac5)
    plt.plot(x, [ac1, ac2, ac3, ac4, ac5], "-x")
    plt.ylim([0.0, 1.20])
    plt.xticks(x)
    plt.xlabel("number of tables in join queries")
    plt.ylabel("average prediction accuracy")
    plt.show()


def plot_times():
    t1 = read_times("1.card", "card-time") * 1000
    t2 = read_times("2.card", "card-time") * 1000
    t3 = read_times("3.card", "card-time") * 1000
    t4 = read_times("4.card", "card-time") * 1000
    t5 = read_times("5.card", "card-time") * 1000
    data = [t1, t2, t3, t4, t5]
    x = range(1, 6)
    plt.boxplot(data, showfliers=False)
    # plt.legend()
    plt.ylabel("latency (ms)")
    plt.xlabel("number of tables in join queries")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    # plot_accuracy()
    plot_times()
