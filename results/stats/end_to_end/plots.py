import matplotlib.pyplot as plt
import numpy as np

from joins.tools import read_from_csv_all
import pandas as pd


def read_times(suffix):
    truth = read_from_csv_all("results/stats/end_to_end/truth"+suffix+".csv")
    deepdb = read_from_csv_all("results/stats/end_to_end/deepdb"+suffix+".csv")
    dhist = read_from_csv_all("results/stats/end_to_end/dhist"+suffix+".csv")
    factorjoin = read_from_csv_all(
        "results/stats/end_to_end/factorjoin"+suffix+".csv")
    flat = read_from_csv_all("results/stats/end_to_end/flat"+suffix+".csv")
    neurocard = read_from_csv_all(
        "results/stats/end_to_end/neurocard"+suffix+".csv")
    wjsample = read_from_csv_all(
        "results/stats/end_to_end/wjsample"+suffix+".csv")
    postgres = read_from_csv_all("results/stats/end_to_end/postgres.csv")

    idx1 = np.array([truth["truth"] != -1][0])
    idx2 = np.array([deepdb["truth"] != -1][0])
    idx3 = np.array([dhist["truth"] != -1][0])
    idx4 = np.array([factorjoin["truth"] != -1][0])
    idx5 = np.array([flat["truth"] != -1][0])
    idx6 = np.array([neurocard["truth"] != -1][0])
    idx7 = np.array([wjsample["truth"] != -1][0])
    idx8 = np.array([postgres["truth"] != -1][0])

    idx = np.where(idx1 & idx2 & idx3 & idx4 & idx5 & idx6 & idx7 & idx8)
    print("total count is ", len(list(idx[0])))

    data = [truth, deepdb, dhist, factorjoin,
            flat, neurocard, wjsample, postgres]
    # print("idx is ", idx)
    # "plan-time"
    execs = []
    plan = []
    for d in data:
        execs.append(np.sum(d["execution-time"].values[idx]))
        plan.append(np.sum(d["plan-time"].values[idx]))
    return execs, plan


def plt_end_to_end():
    execs_10, plan_10 = read_times("_10s")
    execs, plan = read_times("_10s")

    weight_counts_10 = {
        "Plan Time (simple query)": np.array(plan_10)/1000,
        "Execution Time (simple query)": np.array(execs_10)/1000,
    }

    weight_counts = {
        "Plan Time": np.array(plan)/990,
        "Execution Time": np.array(execs)/100,
    }

    width = 0.3
    fig, ax = plt.subplots()

    x = np.array([i for i in range(8)])
    labels = ["TrueCard", "DeepDB", "DHist", "FactorJoin",
              "FLAT", "NeuroCard", "WJSample",  "Postgres"]

    idx = 0
    bottom = np.zeros(8)
    for boolean, weight_count in weight_counts_10.items():
        # print(boolean, weight_count)
        idx += 1
        p = ax.bar(x-0.5*width, weight_count, width,
                   label=boolean, bottom=bottom, alpha=0.3+idx*0.06)
        bottom += weight_count
    bottom = np.zeros(8)
    for boolean, weight_count in weight_counts.items():
        # print(boolean, weight_count)
        p = ax.bar(x+0.5*width, weight_count, width,
                   label=boolean, bottom=bottom)
        bottom += weight_count

    # ax.set_title("Number of penguins with above average body mass")
    ax.legend(loc="center right")
    plt.yscale("log")
    plt.ylim([0.01, 10000])
    plt.xticks(x, labels, rotation=70)
    add_value_labels(ax)

    plt.ylabel("Time (s)")
    plt.xlabel("Method")
    plt.tight_layout()
    plt.show()

    # # data = [5.9, 162, 310, 1.9, 2.7, 0.866]
    # labels = ["BayesCard", "DeepDB", "FLAT",
    #           "FactorJoin", "DHist", "Histogram"]
    # freq_series = pd.Series(data)
    # # plt.bar(data)
    # # plt.xticks(labels)
    # ax = freq_series.plot(
    #     kind="bar", stacked=True, color=["b", "r", "g", "y", "m", "c"], alpha=0.5
    # )
    # # ax.set_title("Amount Frequency")
    # ax.set_xlabel("Method")
    # ax.set_ylabel("Model size (MB)")
    # ax.set_xticklabels(labels)
    # plt.xticks(rotation=0)
    # plt.yscale("log")
    # plt.ylim([0.2, 1000])
    # add_value_labels(ax)
    # plt.tight_layout()
    # plt.show()


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = "bottom"

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = "top"

        if y_value < 1:
            continue
        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha="center",  # Horizontally center label
            va=va,
        )  # Vertically align label differently for
        # positive and negative values.


if __name__ == "__main__":
    plt_end_to_end()
