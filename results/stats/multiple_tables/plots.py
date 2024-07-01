import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from joins.tools import read_from_csv

font = {
    'size': 12}

matplotlib.rc('font', **font)


def plot_accuracy_without_filter():
    truths = read_from_csv("results/stats/multiple_tables/truth.csv", "truth")
    card = read_from_csv("results/stats/multiple_tables/card.csv", "card")
    card_bad = read_from_csv(
        "results/stats/multiple_tables/card_bad.csv", "card")
    idx1 = np.array([truths != -1][0])
    idx2 = np.array([card != -1][0])

    idx = np.where(idx1 & idx2)
    card = card[idx]
    card_bad = card_bad[idx]
    truths = truths[idx]

    re_card = card / truths
    re_card_bad = card_bad / truths

    logbins = np.logspace(
        np.log10(
            min(
                min(re_card),
                min(re_card_bad),
            )
        ),
        np.log10(
            max(
                max(re_card),
                max(re_card_bad),
            )
        ),
        101,
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.hist(re_card_bad, bins=logbins,
             label="DHist-without-jk-correlation", alpha=0.5)
    plt.hist(re_card, bins=logbins,
             label="DHist-with-jk-correlation", alpha=0.5)

    tick = [10 ** (ii) for ii in [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]]
    # print(tick)
    plt.xticks(tick)
    plt.legend()
    plt.ylim([0.7, 1000])
    plt.xlabel("Relative error")
    plt.ylabel("Number of queries")
    plt.show()


def plot_accuracy():
    truths = read_from_csv("results/stats/multiple_tables/truth.csv", "truth")
    card = read_from_csv("results/stats/multiple_tables/card.csv", "card")
    factorjoin = read_from_csv(
        "results/stats/multiple_tables/factorjoin.csv", "factorjoin")
    flat = read_from_csv(
        "workloads/stats_CEB/estimates/stats_CEB_sub_queries_flat.txt", "flat")
    bayescard = read_from_csv(
        "workloads/stats_CEB/estimates/stats_CEB_sub_queries_bayescard.txt", "bayescard")
    # wjsample = read_from_csv("results/stats/multiple_tables/wjsample.csv", "wjsample")
    deepdb = read_from_csv(
        "workloads/stats_CEB/estimates/stats_CEB_sub_queries_deepdb.txt", "deepdb")

    # print([truths != -1][0])
    idx1 = np.array([truths != -1][0])
    idx2 = np.array([card != -1][0])

    idx = np.where(idx1 & idx2)
    # print("idx is ", idx)
    card = card[idx]
    truths = truths[idx]
    deepdb = deepdb[idx]
    flat = flat[idx]
    bayescard = bayescard[idx]
    factorjoin = factorjoin[idx]

    re_card = card / truths
    # re_postgres = postgres / truths
    re_bayescard = bayescard / truths
    # re_wjsample = wjsample / truths
    re_deepdb = deepdb / truths
    re_flat = flat/truths
    re_factorjoin = factorjoin/truths

    fig, axs = plt.subplots(3, 2)

    logbins = np.logspace(
        np.log10(
            min(
                min(re_card),
                min(re_flat),
                min(re_bayescard),
                # min(re_wjsample),
                min(re_deepdb),
                min(re_factorjoin),
            )
        ),
        np.log10(
            max(
                max(re_card),
                max(re_flat),
                max(re_bayescard),
                # max(re_wjsample),
                max(re_deepdb),
                max(re_factorjoin),
            )
        ),
        101,
    )
    # logbins = 301
    # plt.xscale("log")
    axs[0, 0].hist(re_card, bins=logbins, label="DHist")
    axs[0, 0].set_title("DHist")
    axs[1, 0].hist(re_factorjoin, bins=logbins, label="FactorJoin")
    axs[1, 0].set_title("FactorJoin")
    axs[2, 0].hist(re_bayescard, bins=logbins, label="BayesCard")
    axs[2, 0].set_title("BayesCard")
    axs[1, 1].hist(re_flat, bins=logbins, label="FLAT")
    axs[1, 1].set_title("Flat")
    # axs[2, 1].hist(re_wjsample, bins=logbins, label="WJSample")
    axs[2, 1].set_title("WJSample")
    axs[0, 1].hist(re_deepdb, bins=logbins, label="DeepDB")
    axs[0, 1].set_title("DeepDB")
    # # axs[0, 0].legend()

    for ax in axs:
        for a in ax:
            a.set_yscale("log")
            a.set_xscale("log")
            a.set_ylim([0.1, 1000])
            a.set_xticks([0.0001, 0.001, 0.01, 0.1, 1,
                         10, 100, 1000, 10000, 100000])

    fig.text(0.5, 0.01, 'Relative error', ha='center')
    fig.text(0.01, 0.5, 'Number of queries', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


def plot_times():
    truths = read_from_csv(
        "results/stats/single_table/truth.csv", "truth-time-postgres"
    )
    card = read_from_csv("results/stats/single_table/card.csv", "card-time")
    postgres = read_from_csv(
        "results/stats/single_table/postgres.csv", "postgres-time")
    bayescard = read_from_csv(
        "results/stats/single_table/bayescard.csv", "bayescard-time"
    )
    wjsample = read_from_csv(
        "results/stats/single_table/wjsample.csv", "wjsample-time")
    deepdb = read_from_csv(
        "results/stats/single_table/deepdb.csv", "deepdb-time")
    logbins = np.logspace(
        np.log10(
            min(
                min(card),
                min(postgres),
                min(bayescard),
                min(wjsample),
                min(deepdb),
                min(truths),
            )
        ),
        np.log10(
            max(
                max(card),
                max(postgres),
                max(bayescard),
                max(wjsample),
                max(deepdb),
                max(truths),
            )
        ),
        301,
    )
    plt.xscale("log")
    plt.hist(card, bins=logbins, label="card", alpha=0.5)
    plt.hist(postgres, bins=logbins, label="postgres", alpha=0.5)
    plt.hist(bayescard, bins=logbins, label="BayesCard", alpha=0.5)
    plt.hist(wjsample, bins=logbins, label="WJSample", alpha=0.5)
    plt.hist(deepdb, bins=logbins, label="DeepDB", alpha=0.5)
    plt.hist(truths, bins=logbins, label="truth", alpha=0.5)
    plt.legend()
    plt.xlabel("latency (ms)")
    plt.ylabel("# of queries")
    plt.show()


if __name__ == "__main__":
    # plot_accuracy()
    # plot_times()
    plot_accuracy_without_filter()
