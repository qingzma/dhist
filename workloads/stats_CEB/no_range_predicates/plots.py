# python run.py --evaluate --model models/model_stats_200_20.pkl --query workloads/stats_CEB/no_range_predicates/6.sql
# python run.py --evaluate --model models/model_stats_joinhist_200_20.pkl --query workloads/stats_CEB/no_range_predicates/2.sql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joins.tools import read_from_csv, read_from_csv_to_series

plt.figure(dpi=300)


def read_data(first: str, second: str, h1="truth", h2="card"):
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
    truths = [truths1, truths2, truths3, truths4, truths5]

    truths1, jh1 = read_data("1.truth", "1.joinhist")
    truths2, jh2 = read_data("2.truth", "2.joinhist")
    truths3, jh3 = read_data("3.truth", "3.joinhist")
    truths4, jh4 = read_data("4.truth", "4.joinhist")
    truths5, jh5 = read_data("5.truth", "5.joinhist")

    jhs = [jh1, jh2, jh3, jh4, jh5]
    cards = [card1, card2, card3, card4, card5]

    card = [i / j for i, j in zip(cards, truths)]
    jh = [i / j for i, j in zip(jhs, truths)]

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
    plt.axhline(y=1, color="gray", linestyle="--")
    data = [ac1, ac2, ac3, ac4, ac5]
    # plt.plot(x, data, "-x")
    plt.boxplot(
        card,
        showfliers=False,
        showmeans=True,
    )
    for i, j in zip(x, data):
        plt.annotate("%.3f" % j, xy=(i - 0.05, j * 1.03), fontsize=7)
    # plt.ylim([0.0, 1.20])
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
    m1 = [np.mean(i) for i in data]
    st1 = [np.std(i) for i in data]
    x = range(1, 6)
    bp = plt.boxplot(data, showfliers=False)
    for i, line in enumerate(bp["medians"]):
        x, y = line.get_xydata()[1]
        text = " μ={:.2f}\n σ={:.2f}".format(m1[i], st1[i])
        plt.annotate(text, xy=(x, 0.8 * y), fontsize=7)
    # plt.legend()
    plt.ylabel("latency (ms)")
    plt.xlabel("number of tables in join queries")
    plt.yscale("log")
    plt.show()


def plot_2_join_postgres():
    postgres = read_times("2.postgres", "postgres")
    card = read_times("2.card", "card")
    truth = read_times("2.truth", "truth")
    post_error = postgres / truth

    print(np.average(post_error))

    plt.hist(post_error)
    plt.show()


if __name__ == "__main__":
    plot_accuracy()
    # plot_times()
    # plot_2_join_postgres()
