# python run.py --dataset tpcds --evaluate --model models/model_tpcds1g_topk_200_20.pkl --query workloads/tpcds/no_range/4.sql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joins.tools import read_from_csv, read_from_csv_to_series

plt.figure(dpi=200)


def read_data(first: str, second: str, h1="truth", h2="card"):
    truths = read_from_csv_to_series("workloads/tpcds/no_range/" + first + ".csv", h1)
    est = read_from_csv_to_series("workloads/tpcds/no_range/" + second + ".csv", h2)

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
        "workloads/tpcds/no_range/" + first + ".csv", header
    )

    truth1 = truths[~pd.isnull(truths)]
    return truth1.values


def plot_accuracy():
    truths1, card1 = read_data("1_1x_truth", "1_1x_topk_200_20")
    truths2, card2 = read_data("2_1x_truth", "2_1x_topk_200_20")
    truths3, card3 = read_data("3_1x_truth", "3_1x_topk_200_20")
    truths4, card4 = read_data("4_1x_truth", "4_1x_topk_200_20")
    # truths5, card5 = read_data("5_1x_truth", "5_1x_topk_200_20")
    truths = [truths1, truths2, truths3, truths4]

    truths1, jh1 = read_data("1_1x_truth", "1_1x_joinhist_200_20", h2="card")
    truths2, jh2 = read_data("2_1x_truth", "2_1x_joinhist_200_20", h2="card")
    truths3, jh3 = read_data("3_1x_truth", "3_1x_joinhist_200_20", h2="card")
    truths4, jh4 = read_data("4_1x_truth", "4_1x_joinhist_200_20", h2="card")
    # truths5, jh5 = read_data("5_1x_truth", "5_1x_joinhist_200_20", h2="card")

    truths1, ub1 = read_data("1_1x_truth", "1_1x_upperbound_200_20", h2="card")
    truths2, ub2 = read_data("2_1x_truth", "2_1x_upperbound_200_20", h2="card")
    truths3, ub3 = read_data("3_1x_truth", "3_1x_upperbound_200_20", h2="card")
    truths4, ub4 = read_data("4_1x_truth", "4_1x_upperbound_200_20", h2="card")
    # truths5, ub5 = read_data("5_1x_truth", "5_1x_upperbound_200_20", h2="card")

    jhs = [jh1, jh2, jh3, jh4]
    cards = [card1, card2, card3, card4]
    ubs = [ub1, ub2, ub3, ub4]

    card = [i / j for i, j in zip(cards, truths)]
    jh = [i / j for i, j in zip(jhs, truths)]
    ub = [i / j for i, j in zip(ubs, truths)]

    avg_card = [np.average(i) for i in card]
    avg_jh = [np.average(i) for i in jh]
    avg_ub = [np.average(i) for i in ub]

    # print(min(jh[1]), max(ub[1]))
    # print(min(jh[4]), max(ub[4]))

    # postgres = read_from_csv("results/stats/multiple_tables/postgres.csv", "postgres")
    # bayescard = read_from_csv("results/stats/multiple_tables/bayescard.csv", "bayescard")
    # wjsample = read_from_csv("results/stats/multiple_tables/wjsample.csv", "wjsample")
    # deepdb = read_from_csv("results/stats/multiple_tables/deepdb.csv", "deepdb")
    x = range(1, 5)
    # ac1 = np.average(card1 / truths1)
    # ac2 = np.average(card2 / truths2)
    # ac3 = np.average(card3 / truths3)
    # ac4 = np.average(card4 / truths4)
    # ac5 = np.average(card5 / truths5)
    plt.axhline(y=1, color="gray", linestyle="--")
    # data = [ac1, ac2, ac3, ac4, ac5]
    # plt.plot(x, data, "-x")
    bp1 = plt.boxplot(
        card,
        # showfliers=False,
        showmeans=True,
        whis=0,
        medianprops={"linestyle": None, "linewidth": 0},
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=[1, 0, 0, 0.7]),
        flierprops={
            "marker": ".",
            "markersize": 10,
            "markerfacecolor": [1, 0, 0, 0.9],  # "C1",
        },
    )

    bp2 = plt.boxplot(
        jh,
        # showfliers=False,
        showmeans=True,
        whis=0,
        medianprops={"linestyle": None, "linewidth": 0},
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=[0, 1, 0, 0.5]),
        flierprops={
            "marker": ".",
            "markersize": 10,
            "markerfacecolor": [0, 1, 0, 0.8],  # "C7",
        },
    )

    bp3 = plt.boxplot(
        ub,
        # showfliers=False,
        showmeans=True,
        whis=0,
        medianprops={
            "linestyle": None,
            "linewidth": 0,
        },
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor=[0, 0, 1, 0.4]),
        flierprops={
            "marker": ".",
            "markersize": 10,
            "markerfacecolor": [0, 0, 1, 0.8],  # "C3",
        },
    )
    plt.yscale("log")
    plt.legend(
        [bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
        ["DHist", "Join-Histogram", "UpperBound"],
        loc="upper left",
    )

    xxx = 0.1
    yyy = 1.0
    for i, j in zip(x, avg_card):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_jh):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_ub):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    plt.ylim([4e-1, 30])
    plt.xticks(x)
    plt.xlabel("Number of tables in join queries")
    plt.ylabel("prediction accuracy")
    plt.show()


def plot_times():
    t1 = read_times("1.card", "card-time") * 1000
    t2 = read_times("2.card", "card-time") * 1000
    t3 = read_times("3.card", "card-time") * 1000
    t4 = read_times("4.card", "card-time") * 1000
    t5 = read_times("5.card", "card-time") * 1000
    card = [t1, t2, t3, t4, t5]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]

    jh1 = read_times("1.joinhist", "joinhist-time") * 1000
    jh2 = read_times("2.joinhist", "joinhist-time") * 1000
    jh3 = read_times("3.joinhist", "joinhist-time") * 1000
    jh4 = read_times("4.joinhist", "joinhist-time") * 1000
    jh5 = read_times("5.joinhist", "joinhist-time") * 1000

    ub1 = read_times("1.upperbound", "upperbound-time") * 1000
    ub2 = read_times("2.upperbound", "upperbound-time") * 1000
    ub3 = read_times("3.upperbound", "upperbound-time") * 1000
    ub4 = read_times("4.upperbound", "upperbound-time") * 1000
    ub5 = read_times("5.upperbound", "upperbound-time") * 1000

    jh = [jh1, jh2, jh3, jh4, jh5]
    ub = [ub1, ub2, ub3, ub4, ub5]

    m1jh = [np.mean(i) for i in jh]
    st1jh = [np.std(i) for i in jh]
    m1ub = [np.mean(i) for i in ub]
    st1ub = [np.std(i) for i in ub]
    x = range(1, 6)
    plt.plot(x, m1, "-<", label="DHist")
    plt.plot(x, m1jh, "-o", label="Join-Histogram")
    plt.plot(x, m1ub, "-x", label="UpperBound")
    # bp = plt.boxplot(card, showfliers=False)
    # bp_jh = plt.boxplot(jh, showfliers=False)
    # bp_ub = plt.boxplot(ub, showfliers=False)
    # for i, line in enumerate(bp["medians"]):
    #     x, y = line.get_xydata()[1]
    #     text = " μ={:.2f}\n σ={:.2f}".format(m1[i], st1[i])
    #     plt.annotate(text, xy=(x, 0.8 * y), fontsize=7)
    for i, j in zip(x, m1):
        plt.annotate("{:.3f}".format(j), xy=(i, 1.1 * j), fontsize=7)
    for i, j in zip(x, m1jh):
        plt.annotate("{:.3f}".format(j), xy=(i, 1.1 * j), fontsize=7)
    for i, j in zip(x, m1ub):
        plt.annotate("{:.3f}".format(j), xy=(i, 1.1 * j), fontsize=7)
    plt.legend()
    plt.xticks(x)
    plt.ylabel("latency (ms)")
    plt.xlabel("# of tables in join queries")
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


def tune_bin():
    truths1_1x, card1 = read_data("1_1x_truth", "1_1x_topk_200_20", h2="card")
    truths2_1x, card2 = read_data("2_1x_truth", "2_1x_topk_200_20", h2="card")
    truths3_1x, card3 = read_data("3_1x_truth", "3_1x_topk_200_20", h2="card")
    truths4_1x, card4 = read_data("4_1x_truth", "4_1x_topk_200_20", h2="card")
    # truths5, card5 = read_data(_1x"5_truth", "5_1x_topk_20_10", h2="card")

    truths1_1x, jh1 = read_data("1_1x_truth", "1_1x_topk_100_10", h2="card")
    truths2_1x, jh2 = read_data("2_1x_truth", "2_1x_topk_100_10", h2="card")
    truths3_1x, jh3 = read_data("3_1x_truth", "3_1x_topk_100_10", h2="card")
    truths4_1x, jh4 = read_data("4_1x_truth", "4_1x_topk_100_10", h2="card")
    # truths5, jh5 = read_data("5_1x_truth", "5_1x_topk_100_10", h2="card")

    truths1_10x, ub1 = read_data("1_1x_truth", "1_10x_topk_200_20", h2="card")
    truths2_10x, ub2 = read_data("2_1x_truth", "2_10x_topk_200_20", h2="card")
    truths3_10x, ub3 = read_data("3_1x_truth", "3_10x_topk_200_20", h2="card")
    truths4_10x, ub4 = read_data("4_1x_truth", "4_10x_topk_200_20", h2="card")
    # truths5, ub5 = read_data("5_10x_truth", "5_10x_topk_200_20", h2="card")

    truths1_10x, b200_1 = read_data("1_10x_truth", "1_10x_topk_100_10", h2="card")
    truths2_10x, b200_2 = read_data("2_1x_truth", "2_10x_topk_100_10", h2="card")
    truths3_10x, b200_3 = read_data("3_1x_truth", "3_10x_topk_100_10", h2="card")
    truths4_10x, b200_4 = read_data("4_1x_truth", "4_10x_topk_100_10", h2="card")

    truths_1x = [truths1_1x, truths2_1x, truths3_1x, truths4_1x]
    truths_10x = [truths1_10x, truths2_10x, truths3_10x, truths4_10x]
    jhs = [jh1, jh2, jh3, jh4]
    cards = [card1, card2, card3, card4]
    ubs = [ub1, ub2, ub3, ub4]
    b200s = [b200_1, b200_2, b200_3, b200_4]
    # b400s = [b400_1, b400_2, b400_3, b400_4, b400_5]

    card = [i / j for i, j in zip(cards, truths_1x)]
    jh = [i / j for i, j in zip(jhs, truths_1x)]
    ub = [i / j for i, j in zip(ubs, truths_10x)]
    b200 = [i / j for i, j in zip(b200s, truths_10x)]
    # b400 = [i / j for i, j in zip(b400s, truths)]

    avg_card = [np.average(i) for i in card]
    avg_jh = [np.average(i) for i in jh]
    avg_ub = [np.average(i) for i in ub]
    avg_b200 = [np.average(i) for i in b200]
    # avg_b400 = [np.average(i) for i in b400]

    x = range(1, 5)

    plt.axhline(y=1, color="gray", linestyle="--")

    plt.plot(x, avg_card, "-x", label="Accuracy: bin=200, k=20")
    plt.plot(x, avg_jh, "-o", label="Accuracy: bin=100, k=10")
    # plt.plot(x, avg_ub, "-h", label="10x 200")
    # plt.plot(x, avg_b200, "-P", label="10x 100")
    # plt.plot(x, avg_b400, "-P", label="bin size = 400")

    plt.legend()
    # plt.yscale("log")
    xxx = 0.1
    yyy = 1.0
    for i, j in zip(x, avg_card):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_jh):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_ub):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_b200):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b400):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    plt.ylim([0.5, 1.2])
    plt.xticks(x)
    plt.xlabel("Number of tables in join queries")
    plt.ylabel("prediction accuracy")
    plt.show()


def tune_bin_times():
    t1 = read_times("1_1x_topk_200_20", "card-time") * 1000
    t2 = read_times("2_1x_topk_200_20", "card-time") * 1000
    t3 = read_times("3_1x_topk_200_20", "card-time") * 1000
    t4 = read_times("4_1x_topk_200_20", "card-time") * 1000
    # t5 = read_times("5_1x_topk_200_20", "card-time") * 1000
    card = [t1, t2, t3, t4]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 5)
    plt.plot(x, m1, "-x", label="Latency: bin=200, k=20")

    t1 = read_times("1_1x_topk_100_10", "card-time") * 1000
    t2 = read_times("2_1x_topk_100_10", "card-time") * 1000
    t3 = read_times("3_1x_topk_100_10", "card-time") * 1000
    t4 = read_times("4_1x_topk_100_10", "card-time") * 1000
    # t5 = read_times("5_1x_topk_100_10", "card-time") * 1000
    card = [t1, t2, t3, t4]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 5)
    plt.plot(x, m1, "-o", label="Latency: bin=100, k=10")
    x = range(1, 5)

    # plt.axhline(y=1, color="gray", linestyle="--")

    plt.legend()
    plt.yscale("log")
    # xxx = 0.1
    # yyy = 1.0
    # for i, j in zip(x, avg_card):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_jh):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_ub):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b200):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b400):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # plt.ylim([0.5, 1.2])
    plt.xticks(x)
    plt.xlabel("Number of tables in join queries")
    plt.ylabel("latency (ms)")
    plt.show()


def tune_bin_mixed():
    truths1_1x, card1 = read_data("1_1x_truth", "1_1x_topk_200_20", h2="card")
    truths2_1x, card2 = read_data("2_1x_truth", "2_1x_topk_200_20", h2="card")
    truths3_1x, card3 = read_data("3_1x_truth", "3_1x_topk_200_20", h2="card")
    truths4_1x, card4 = read_data("4_1x_truth", "4_1x_topk_200_20", h2="card")
    # truths5, card5 = read_data(_1x"5_truth", "5_1x_topk_20_10", h2="card")

    truths1_1x, jh1 = read_data("1_1x_truth", "1_1x_topk_100_10", h2="card")
    truths2_1x, jh2 = read_data("2_1x_truth", "2_1x_topk_100_10", h2="card")
    truths3_1x, jh3 = read_data("3_1x_truth", "3_1x_topk_100_10", h2="card")
    truths4_1x, jh4 = read_data("4_1x_truth", "4_1x_topk_100_10", h2="card")
    # truths5, jh5 = read_data("5_1x_truth", "5_1x_topk_100_10", h2="card")

    truths1_10x, ub1 = read_data("1_1x_truth", "1_10x_topk_200_20", h2="card")
    truths2_10x, ub2 = read_data("2_1x_truth", "2_10x_topk_200_20", h2="card")
    truths3_10x, ub3 = read_data("3_1x_truth", "3_10x_topk_200_20", h2="card")
    truths4_10x, ub4 = read_data("4_1x_truth", "4_10x_topk_200_20", h2="card")
    # truths5, ub5 = read_data("5_10x_truth", "5_10x_topk_200_20", h2="card")

    truths1_10x, b200_1 = read_data("1_10x_truth", "1_10x_topk_100_10", h2="card")
    truths2_10x, b200_2 = read_data("2_1x_truth", "2_10x_topk_100_10", h2="card")
    truths3_10x, b200_3 = read_data("3_1x_truth", "3_10x_topk_100_10", h2="card")
    truths4_10x, b200_4 = read_data("4_1x_truth", "4_10x_topk_100_10", h2="card")

    truths_1x = [truths1_1x, truths2_1x, truths3_1x, truths4_1x]
    truths_10x = [truths1_10x, truths2_10x, truths3_10x, truths4_10x]
    jhs = [jh1, jh2, jh3, jh4]
    cards = [card1, card2, card3, card4]
    ubs = [ub1, ub2, ub3, ub4]
    b200s = [b200_1, b200_2, b200_3, b200_4]
    # b400s = [b400_1, b400_2, b400_3, b400_4, b400_5]

    card = [i / j for i, j in zip(cards, truths_1x)]
    jh = [i / j for i, j in zip(jhs, truths_1x)]
    ub = [i / j for i, j in zip(ubs, truths_10x)]
    b200 = [i / j for i, j in zip(b200s, truths_10x)]
    # b400 = [i / j for i, j in zip(b400s, truths)]

    avg_card = [np.average(i) for i in card]
    avg_jh = [np.average(i) for i in jh]
    avg_ub = [np.average(i) for i in ub]
    avg_b200 = [np.average(i) for i in b200]
    # avg_b400 = [np.average(i) for i in b400]

    x = range(1, 5)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.axhline(y=1, color="gray", linestyle="--")

    ax1.plot(x, avg_card, "-x", label="Accuracy: bin=200, k=20")
    ax1.plot(x, avg_jh, "-o", label="Accuracy: bin=100, k=10")
    # plt.plot(x, avg_ub, "-h", label="10x 200")
    # plt.plot(x, avg_b200, "-P", label="10x 100")
    # plt.plot(x, avg_b400, "-P", label="bin size = 400")

    ax1.legend()
    # plt.yscale("log")
    xxx = 0.1
    yyy = 1.0
    for i, j in zip(x, avg_card):
        ax1.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_jh):
        ax1.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_ub):
        ax1.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_b200):
        ax1.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b400):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    ax1.set_ylim([0.5, 1.2])
    ax1.set_xticks(x)
    ax1.set_xlabel("Number of tables in join queries")
    ax1.set_ylabel("prediction accuracy")
    # plt.show()

    t1 = read_times("1_1x_topk_200_20", "card-time") * 1000
    t2 = read_times("2_1x_topk_200_20", "card-time") * 1000
    t3 = read_times("3_1x_topk_200_20", "card-time") * 1000
    t4 = read_times("4_1x_topk_200_20", "card-time") * 1000
    # t5 = read_times("5_1x_topk_200_20", "card-time") * 1000
    card = [t1, t2, t3, t4]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 5)
    ax2.plot(x, m1, "-.P", label="Latency: bin=200, k=20", color="k")

    t1 = read_times("1_1x_topk_100_10", "card-time") * 1000
    t2 = read_times("2_1x_topk_100_10", "card-time") * 1000
    t3 = read_times("3_1x_topk_100_10", "card-time") * 1000
    t4 = read_times("4_1x_topk_100_10", "card-time") * 1000
    # t5 = read_times("5_1x_topk_100_10", "card-time") * 1000
    card = [t1, t2, t3, t4]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 5)
    ax2.plot(x, m1, "--^", label="Latency: bin=100, k=10", color="c")
    x = range(1, 5)

    # plt.axhline(y=1, color="gray", linestyle="--")

    ax2.legend(loc="lower right")
    ax2.set_yscale("log")
    # xxx = 0.1
    # yyy = 1.0
    # for i, j in zip(x, avg_card):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_jh):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_ub):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b200):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b400):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # plt.ylim([0.5, 1.2])
    # plt.xticks(x)
    # plt.xlabel("Number of tables in join queries")
    ax2.set_ylabel("latency (ms)")
    ax2.set_ylim([0.01, 10000])
    plt.tight_layout()
    plt.show()


def tune_k():
    truths1, card1 = read_data("1.truth", "1.topk_100_1", h2="card")
    truths2, card2 = read_data("2.truth", "2.topk_100_1", h2="card")
    truths3, card3 = read_data("3.truth", "3.topk_100_1", h2="card")
    truths4, card4 = read_data("4.truth", "4.topk_100_1", h2="card")
    truths5, card5 = read_data("5.truth", "5.topk_100_1", h2="card")
    truths = [truths1, truths2, truths3, truths4, truths5]

    truths1, jh1 = read_data("1.truth", "1.topk_100_5", h2="card")
    truths2, jh2 = read_data("2.truth", "2.topk_100_5", h2="card")
    truths3, jh3 = read_data("3.truth", "3.topk_100_5", h2="card")
    truths4, jh4 = read_data("4.truth", "4.topk_100_5", h2="card")
    truths5, jh5 = read_data("5.truth", "5.topk_100_5", h2="card")

    truths1, ub1 = read_data("1.truth", "1.topk_100_10", h2="card")
    truths2, ub2 = read_data("2.truth", "2.topk_100_10", h2="card")
    truths3, ub3 = read_data("3.truth", "3.topk_100_10", h2="card")
    truths4, ub4 = read_data("4.truth", "4.topk_100_10", h2="card")
    truths5, ub5 = read_data("5.truth", "5.topk_100_10", h2="card")

    truths1, b200_1 = read_data("1.truth", "1.topk_100_20", h2="card")
    truths2, b200_2 = read_data("2.truth", "2.topk_100_20", h2="card")
    truths3, b200_3 = read_data("3.truth", "3.topk_100_20", h2="card")
    truths4, b200_4 = read_data("4.truth", "4.topk_100_20", h2="card")
    truths5, b200_5 = read_data("5.truth", "5.topk_100_20", h2="card")

    truths1, b400_1 = read_data("1.truth", "1.topk_100_40", h2="card")
    truths2, b400_2 = read_data("2.truth", "2.topk_100_40", h2="card")
    truths3, b400_3 = read_data("3.truth", "3.topk_100_40", h2="card")
    truths4, b400_4 = read_data("4.truth", "4.topk_100_40", h2="card")
    truths5, b400_5 = read_data("5.truth", "5.topk_100_40", h2="card")

    jhs = [jh1, jh2, jh3, jh4, jh5]
    cards = [card1, card2, card3, card4, card5]
    ubs = [ub1, ub2, ub3, ub4, ub5]
    b200s = [b200_1, b200_2, b200_3, b200_4, b200_5]
    b400s = [b400_1, b400_2, b400_3, b400_4, b400_5]

    card = [i / j for i, j in zip(cards, truths)]
    jh = [i / j for i, j in zip(jhs, truths)]
    ub = [i / j for i, j in zip(ubs, truths)]
    b200 = [i / j for i, j in zip(b200s, truths)]
    b400 = [i / j for i, j in zip(b400s, truths)]

    avg_card = [np.average(i) for i in card]
    avg_jh = [np.average(i) for i in jh]
    avg_ub = [np.average(i) for i in ub]
    avg_b200 = [np.average(i) for i in b200]
    avg_b400 = [np.average(i) for i in b400]

    x = range(1, 6)

    plt.axhline(y=1, color="gray", linestyle="--")

    plt.plot(x, avg_card, "-x", label="k = 1")
    plt.plot(x, avg_jh, "-o", label="k = 5")
    plt.plot(x, avg_ub, "-h", label="k = 10")
    plt.plot(x, avg_b200, "-P", label="k = 20")
    plt.plot(x, avg_b400, "-P", label="k = 40")

    plt.legend()
    # plt.yscale("log")
    xxx = 0.1
    yyy = 1.0
    for i, j in zip(x, avg_card):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_jh):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_ub):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_b200):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    for i, j in zip(x, avg_b400):
        plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # plt.ylim([0.5, 1.2])
    plt.xticks(x)
    plt.xlabel("# of tables in join queries")
    plt.ylabel("prediction accuracy")
    plt.show()


def tune_k_times():
    t1 = read_times("1.topk_100_1", "card-time") * 1000
    t2 = read_times("2.topk_100_1", "card-time") * 1000
    t3 = read_times("3.topk_100_1", "card-time") * 1000
    t4 = read_times("4.topk_100_1", "card-time") * 1000
    t5 = read_times("5.topk_100_1", "card-time") * 1000
    card = [t1, t2, t3, t4, t5]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 6)
    plt.plot(x, m1, "-x", label="k = 1")

    t1 = read_times("1.topk_100_5", "card-time") * 1000
    t2 = read_times("2.topk_100_5", "card-time") * 1000
    t3 = read_times("3.topk_100_5", "card-time") * 1000
    t4 = read_times("4.topk_100_5", "card-time") * 1000
    t5 = read_times("5.topk_100_5", "card-time") * 1000
    card = [t1, t2, t3, t4, t5]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 6)
    plt.plot(x, m1, "-o", label="k = 5")

    t1 = read_times("1.topk_100_10", "card-time") * 1000
    t2 = read_times("2.topk_100_10", "card-time") * 1000
    t3 = read_times("3.topk_100_10", "card-time") * 1000
    t4 = read_times("4.topk_100_10", "card-time") * 1000
    t5 = read_times("5.topk_100_10", "card-time") * 1000
    card = [t1, t2, t3, t4, t5]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 6)
    plt.plot(x, m1, "-h", label="k = 10")

    t1 = read_times("1.topk_100_20", "card-time") * 1000
    t2 = read_times("2.topk_100_20", "card-time") * 1000
    t3 = read_times("3.topk_100_20", "card-time") * 1000
    t4 = read_times("4.topk_100_20", "card-time") * 1000
    t5 = read_times("5.topk_100_20", "card-time") * 1000
    card = [t1, t2, t3, t4, t5]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 6)
    plt.plot(x, m1, "-p", label="k = 20")

    t1 = read_times("1.topk_100_40", "card-time") * 1000
    t2 = read_times("2.topk_100_40", "card-time") * 1000
    t3 = read_times("3.topk_100_40", "card-time") * 1000
    t4 = read_times("4.topk_100_40", "card-time") * 1000
    t5 = read_times("5.topk_100_40", "card-time") * 1000
    card = [t1, t2, t3, t4, t5]
    m1 = [np.mean(i) for i in card]
    st1 = [np.std(i) for i in card]
    x = range(1, 6)
    plt.plot(x, m1, "-P", label="k = 40")

    x = range(1, 6)

    # plt.axhline(y=1, color="gray", linestyle="--")

    plt.legend()
    # plt.yscale("log")
    # xxx = 0.1
    # yyy = 1.0
    # for i, j in zip(x, avg_card):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_jh):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_ub):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b200):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # for i, j in zip(x, avg_b400):
    #     plt.annotate("%.3f" % j, xy=(i - xxx, j * yyy), fontsize=7)
    # plt.ylim([0.5, 1.2])
    plt.xticks(x)
    plt.xlabel("# of tables in join queries")
    plt.ylabel("latency (ms)")
    plt.show()


def plot_bin_model_size():
    x = [20, 50, 100, 200, 400]
    size = np.array([360726, 790214, 1431741, 2278090, 3151738])
    size = size / (1024.0 * 1024)
    plt.plot(x, size, "-o")
    plt.ylabel("size (MB)")
    plt.xlabel("bin size")
    plt.xticks(x)
    plt.show()


def plot_k_model_size():
    x = [1, 5, 10, 20, 40]
    size = np.array([429412, 899000, 1431741, 2063374, 2456373])
    size = size / (1024.0 * 1024)
    plt.plot(x, size, "-o")
    plt.ylabel("size (MB)")
    plt.xlabel("k value")
    plt.xticks(x)
    plt.show()


if __name__ == "__main__":
    # plot_accuracy()
    # tune_bin()
    # tune_bin_times()
    tune_bin_mixed()
