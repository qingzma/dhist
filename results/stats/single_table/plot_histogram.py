import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joins.tools import read_from_csv

fig = plt.figure(figsize=(4, 2), dpi=100)


def plot_1d_histogram(file_path, col_header):
    data = pd.read_csv(file_path)[col_header].values
    plt.xlabel(col_header)
    plt.ylabel("frequency of UserId")
    plt.hist(data, 300, alpha=0.9, label=file_path.split("/")[2].split(".")[0])
    fig.tight_layout()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    # plt.yscale("log")
    # plt.show()


if __name__ == "__main__":
    plot_1d_histogram("data/stats/comments.csv", "UserId")
    plot_1d_histogram("data/stats/badges.csv", "UserId")

    # plot_1d_histogram("data/stats/users.csv", "Id")
    plt.legend()
    plt.show()

    # plot_times()
