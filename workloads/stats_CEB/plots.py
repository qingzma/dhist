import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from joins.tools import read_from_csv, read_from_csv_to_series

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


# function to add value labels
def addlabels(x, y, ax, color):
    y = 100 * y  # , transform=ax.transAxes
    for i in range(len(x)):
        plt.text(
            0.95 * (i + 1), 0.6 * y[i], "{:.2f}%".format(y[i]), color=color
        )  # , transform=ax.transAxes


def plot_dominating_trend():
    fig, ax = plt.subplots()
    fig.set_dpi(250)
    x1 = np.array(
        [
            1.0 / 40325,
            456.0 / 79851,
            6028320.0 / 15900001,
            1427299116.0 / 2810041173,
        ]
    )
    x10 = np.array(
        [
            10.0 / 40325,
            2316.0 / 79851,
            12207751.0 / 15900001,
            2703632552.0 / 2810041173,
        ]
    )
    x100 = np.array(
        [
            100.0 / 40325,
            7953.0 / 79851,
            14874968.0 / 15900001,
            2800877453.0 / 2810041173,
        ]
    )
    x1000 = np.array(
        [
            1000.0 / 40325,
            23463.0 / 79851,
            15702975.0 / 15900001,
            2809920313.0 / 2810041173,
        ]
    )

    x = [1, 2, 3, 4]

    ax.plot(x, 100 * x1, "-o", label="k=1", color=colors[0])
    ax.plot(x, 100 * x10, "-v", label="k=10", color=colors[1])
    ax.plot(x, 100 * x100, "-1", label="k=100", color=colors[2])
    ax.plot(x, 100 * x1000, "-s", label="k=1000", color=colors[3])
    ax.set_xticks(x)
    ax.legend()

    addlabels(x, x1, ax, color=colors[0])
    addlabels(x, x10, ax, color=colors[1])
    addlabels(x[:-2], x100, ax, color=colors[2])
    addlabels(x[:-2], x1000, ax, color=colors[3])
    plt.yscale("log")

    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax.set_xlabel("# of tables in a join query")
    ax.set_ylabel("proportion of top k join paths in join results (%)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_dominating_trend()
