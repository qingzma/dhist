import matplotlib.pyplot as plt
import numpy as np

from joins.tools import read_from_csv


def plot_accuracy():
    truths = read_from_csv("results/stats/single_table/truth.csv", "truth")
    card = read_from_csv("results/stats/single_table/card.csv", "card")
    postgres = read_from_csv("results/stats/single_table/postgres.csv", "postgres")
    bayescard = read_from_csv("results/stats/single_table/bayescard.csv", "bayescard")
    wjsample = read_from_csv("results/stats/single_table/wjsample.csv", "wjsample")
    deepdb = read_from_csv("results/stats/single_table/deepdb.csv", "deepdb")

    re_card = card / truths
    re_postgres = postgres / truths
    re_bayescard = bayescard / truths
    re_wjsample = wjsample / truths
    re_deepdb = deepdb / truths

    fig, axs = plt.subplots(3, 2)

    logbins = np.logspace(
        np.log10(
            min(
                min(re_card),
                min(re_postgres),
                min(re_bayescard),
                min(re_wjsample),
                min(re_deepdb),
            )
        ),
        np.log10(
            max(
                max(re_card),
                max(re_postgres),
                max(re_bayescard),
                max(re_wjsample),
                max(re_deepdb),
            )
        ),
        51,
    )
    # logbins = 301
    plt.xscale("log")
    axs[0, 0].hist(re_card, bins=logbins, label="card")
    axs[0, 0].set_title("card")
    axs[1, 0].hist(re_postgres, bins=logbins, label="postgres")
    axs[1, 0].set_title("postgres")
    axs[2, 0].hist(re_bayescard, bins=logbins, label="BayesCard")
    axs[2, 0].set_title("BayesCard")
    axs[0, 1].hist(re_wjsample, bins=logbins, label="WJSample")
    axs[0, 1].set_title("WJSample")
    axs[1, 1].hist(re_deepdb, bins=logbins, label="DeepDB")
    axs[1, 1].set_title("DeepDB")
    # axs[0, 0].legend()

    for ax in axs:
        for a in ax:
            a.set_yscale("log")
            a.set_xscale("log")
            a.set_xlim([0.1, 1000])
    plt.show()


def plot_times():
    truths = read_from_csv(
        "results/stats/single_table/truth.csv", "truth-time-postgres"
    )
    card = read_from_csv("results/stats/single_table/card.csv", "card-time")
    postgres = read_from_csv("results/stats/single_table/postgres.csv", "postgres-time")
    bayescard = read_from_csv(
        "results/stats/single_table/bayescard.csv", "bayescard-time"
    )
    wjsample = read_from_csv("results/stats/single_table/wjsample.csv", "wjsample-time")
    deepdb = read_from_csv("results/stats/single_table/deepdb.csv", "deepdb-time")
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
    plot_accuracy()
    # plot_times()
