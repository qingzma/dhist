import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joins.tools import read_from_csv


def plot_1d_histogram(file_path, col_header):
    data = pd.read_csv(file_path)[col_header].values
    plt.xlabel(col_header)
    plt.ylabel("number of points")
    plt.hist(data, 300)
    plt.show()


if __name__ == "__main__":
    # plot_1d_histogram("data/stats/badges.csv", "UserId")
    # plot_1d_histogram("data/stats/comments.csv", "UserId")
    plot_1d_histogram("data/stats/users.csv", "Id")

    # plot_times()
