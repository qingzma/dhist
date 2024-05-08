import matplotlib.pyplot as plt
import numpy as np

from joins.tools import read_from_csv


def plot_accuracy():
    truths = read_from_csv("results/stats/single_table/truth.csv", "truth")
    card = read_from_csv("results/stats/single_table/card.csv", "card")
    postgres = read_from_csv("results/stats/single_table/postgres.csv", "postgres")
    re_card = card / truths
    re_postgres = postgres / truths
    logbins = np.logspace(
        np.log10(min(min(re_card), min(re_postgres))),
        np.log10(max(max(re_card), max(re_postgres))),
        301,
    )
    plt.xscale("log")
    plt.hist(re_card, bins=logbins, label="card", alpha=0.5)
    plt.hist(re_postgres, bins=logbins, label="postgres", alpha=0.5)
    plt.legend()
    plt.show()


def plot_times():
    truths = read_from_csv(
        "results/stats/single_table/truth.csv", "truth-time-postgres"
    )
    card = read_from_csv("results/stats/single_table/card.csv", "card-time") * 1000
    postgres = read_from_csv("results/stats/single_table/postgres.csv", "postgres-time")
    logbins = np.logspace(
        np.log10(min(min(card), min(postgres), min(truths))),
        np.log10(max(max(card), max(postgres), max(truths))),
        301,
    )
    plt.xscale("log")
    plt.hist(card, bins=logbins, label="card", alpha=0.5)
    plt.hist(postgres, bins=logbins, label="postgres", alpha=0.5)
    plt.hist(truths, bins=logbins, label="truth", alpha=0.5)
    plt.legend()
    plt.xlabel("latency (ms)")
    plt.ylabel("# of queries")
    plt.show()


if __name__ == "__main__":
    # plot_accuracy()
    plot_times()
