import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joins.tools import read_from_csv_all


def read_times(suffix):
    truth = read_from_csv_all("results/stats/end_to_end/truth_machine.csv")
    deepdb = read_from_csv_all("results/stats/end_to_end/deepdb" + suffix + ".csv")
    dhist = read_from_csv_all("results/stats/end_to_end/dhist" + suffix + ".csv")
    factorjoin = read_from_csv_all(
        "results/stats/end_to_end/factorjoin" + suffix + ".csv"
    )
    flat = read_from_csv_all("results/stats/end_to_end/flat" + suffix + ".csv")
    neurocard = read_from_csv_all(
        "results/stats/end_to_end/neurocard" + suffix + ".csv"
    )
    wjsample = read_from_csv_all("results/stats/end_to_end/wjsample" + suffix + ".csv")
    postgres = read_from_csv_all("results/stats/end_to_end/postgres_machine.csv")

    idx1 = np.array([truth["truth"] != -1][0])
    idx2 = np.array([deepdb["truth"] != -1][0])
    idx3 = np.array([dhist["truth"] != -1][0])
    idx4 = np.array([factorjoin["truth"] != -1][0])
    idx5 = np.array([flat["truth"] != -1][0])
    idx6 = np.array([neurocard["truth"] != -1][0])
    idx7 = np.array([wjsample["truth"] != -1][0])
    idx8 = np.array([postgres["truth"] != -1][0])
    # # print("idx1", idx1)
    # print("idx1", len(idx1))
    # print("idx2", len(idx2))
    # print("idx3", len(idx3))
    # print("idx4", len(idx4))
    # print("idx5", len(idx5))
    # print("idx6", len(idx6))
    # print("idx7", len(idx7))
    # print("idx8", len(idx8))

    idx = np.where(idx1 & idx2 & idx3 & idx4 & idx5 & idx6 & idx7 & idx8)
    print("total count is ", len(list(idx[0])))

    data = [truth, deepdb, dhist, factorjoin, flat, neurocard, wjsample, postgres]
    # print("idx is ", idx)
    # "plan-time"
    execs = []
    plan = []
    for d in data:
        execs.append(np.sum(d["execution-time"].values[idx]))
        plan.append(np.sum(d["plan-time"].values[idx]))
    return execs, plan


def read_times_all():
    truth = read_from_csv_all("results/stats/end_to_end/truth_machine.csv")
    deepdb = read_from_csv_all("results/stats/end_to_end/deepdb_600s.csv")
    dhist = read_from_csv_all("results/stats/end_to_end/dhist_machine.csv")
    factorjoin = read_from_csv_all("results/stats/end_to_end/factorjoin_machine.csv")
    flat = read_from_csv_all("results/stats/end_to_end/flat_600s.csv")
    neurocard = read_from_csv_all("results/stats/end_to_end/neurocard_machine.csv")
    wjsample = read_from_csv_all("results/stats/end_to_end/wjsample_600s.csv")
    postgres = read_from_csv_all("results/stats/end_to_end/postgres_machine.csv")

    idx1 = np.array([truth["truth"] != -1][0])
    idx2 = np.array([deepdb["truth"] != -1][0])
    idx3 = np.array([dhist["truth"] != -1][0])
    idx4 = np.array([factorjoin["truth"] != -1][0])
    idx5 = np.array([flat["truth"] != -1][0])
    idx6 = np.array([neurocard["truth"] != -1][0])
    idx7 = np.array([wjsample["truth"] != -1][0])
    idx8 = np.array([postgres["truth"] != -1][0])
    # # print("idx1", idx1)
    # print("idx1", len(idx1))
    # print("idx2", len(idx2))
    # print("idx3", len(idx3))
    # print("idx4", len(idx4))
    # print("idx5", len(idx5))
    # print("idx6", len(idx6))
    # print("idx7", len(idx7))
    # print("idx8", len(idx8))

    idx = np.where(idx1 & idx2 & idx3 & idx4 & idx5 & idx6 & idx7 & idx8)
    # print("total count is ", len(list(idx[0])))

    data = [truth, deepdb, dhist, factorjoin, flat, neurocard, wjsample, postgres]
    # print("idx is ", idx)
    # "plan-time"
    execs = []
    plan = []
    cnt_10ms = []
    cnt_100ms = []
    cnt_1 = []
    cnt_10 = []
    cnt_100 = []
    cnt_600 = []

    # compensate for deepdb, wjsample,etc.
    compensate = [
        (1499565 + 961267 + 932148) * 1.2,
        (95707 + 1499565 + 961267 + 932148) * 1.2,
    ]
    for d in data:
        # execs.append(np.sum(d["execution-time"].values[idx]))
        # plan.append(np.sum(d["plan-time"].values[idx]))
        cnt_fail = len([i for i in d["truth"].values if i == -1])
        val = np.sum(d["execution-time"].values)
        if cnt_fail > 0:
            print("failed ", cnt_fail, " times.")
            val += compensate[cnt_fail - 3]
        execs.append(val)
        plan.append(np.sum(d["plan-time"].values))
        cnt_10ms.append(
            len([i for i in d["execution-time"].values if i < 50 and i != 0])
        )
        cnt_100ms.append(
            len([i for i in d["execution-time"].values if i < 100 and i != 0])
        )
        cnt_1.append(
            len([i for i in d["execution-time"].values if i < 1000 and i != 0])
        )
        cnt_10.append(
            len([i for i in d["execution-time"].values if i < 10000 and i != 0])
        )
        cnt_100.append(
            len([i for i in d["execution-time"].values if i < 100000 and i != 0])
        )
        cnt_600.append(
            len([i for i in d["execution-time"].values if i < 600000 and i != 0])
        )
    print(cnt_10ms)
    print(cnt_100ms)
    print(cnt_1)
    print(cnt_10)
    print(cnt_100)
    print(cnt_600)

    plan_compensate_from_card_estimation = [
        24,
        152,
        34,
        32,
        411,
        52,
        41,
        27,
    ]
    # print("plan is ", plan)
    for i in range(8):
        plan[i] += plan_compensate_from_card_estimation[i] * 1000
    # print("plan is ", plan)

    return execs, plan, cnt_10ms, cnt_100ms, cnt_1, cnt_10, cnt_100


def plt_end_to_end_all():
    execs, plan, cnt_50ms, cnt_100ms, cnt_1, cnt_10, cnt_100 = read_times_all()

    weight_counts = {
        "Plan Time": np.array(plan) / 1000,
        "Execution Time": np.array(execs) / 1000,
    }

    width = 0.4
    fig, ax = plt.subplots()

    x = np.array([i for i in range(8)])
    labels = [
        "TrueCard",
        "DeepDB",
        "DHist",
        "FactorJoin",
        "FLAT",
        "NeuroCard",
        "WJSample",
        "Postgres",
    ]

    idx = 0
    # bottom = np.zeros(8)
    # for boolean, weight_count in weight_counts_10.items():
    #     # print(boolean, weight_count)
    #     idx += 1
    #     p = ax.bar(
    #         x - 0.5 * width,
    #         weight_count,
    #         width,
    #         label=boolean,
    #         bottom=bottom,
    #         alpha=0.3 + idx * 0.06,
    #     )
    #     bottom += weight_count
    bottom = np.zeros(8)
    for boolean, weight_count in weight_counts.items():
        # print(boolean, weight_count)
        p = ax.bar(x, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    # ax.set_title("Number of penguins with above average body mass")
    ax.legend(loc="upper right")
    plt.yscale("log")
    plt.ylim([10, 4e4])
    plt.xticks(x, labels, rotation=70)
    # add_value_labels(ax)
    # w_10_1 = weight_counts_10["Plan Time (simple query)"]
    # w_10_2 = w_10_1 + weight_counts_10["Execution Time (simple query)"]
    w_1 = weight_counts["Plan Time"]
    w_2 = w_1 + weight_counts["Execution Time"]
    for i in range(8):

        # ax.annotate(
        #     "{:.2f}".format(w_1[i]),  # Use `label` as label
        #     (i + 0.5 * width, w_1[i]),  # Place label at end of the bar
        #     ha="center",  # Horizontally center label
        # )
        ax.annotate(
            "{:.0f}".format(w_2[i]),  # Use `label` as label
            (i, w_2[i]),  # Place label at end of the bar
            ha="center",  # Horizontally center label
        )

    plt.ylabel("Total time (s)")
    # plt.xlabel("Method")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    plt.bar(labels, cnt_100, hatch="/", label="<100s")
    plt.bar(labels, cnt_10, hatch="\\", label="<10s")
    plt.bar(labels, cnt_1, hatch="x", label="<1s")
    plt.bar(labels, cnt_100ms, hatch="--", label="<100ms")
    plt.bar(labels, cnt_50ms, hatch="+", label="<50ms")

    values = [cnt_50ms, cnt_100ms, cnt_1, cnt_10, cnt_100]
    for i in range(8):
        for value in values:
            ax.annotate(
                "{:.0f}".format(value[i]),  # Use `label` as label
                (i, value[i]),  # Place label at end of the bar
                ha="center",  # Horizontally center label
                weight="bold",
            )

    ax.legend(loc="upper right", ncol=5)
    # plt.yscale("log")
    plt.ylim([1, 170])
    plt.xticks(x, labels, rotation=70)
    ax.set_ylabel("Number of queries")
    plt.tight_layout()
    plt.show()


def plt_end_to_end():
    execs_10, plan_10 = read_times("_10s")
    execs, plan = read_times("")

    weight_counts_10 = {
        "Plan Time (simple query)": np.array(plan_10) / 1000,
        "Execution Time (simple query)": np.array(execs_10) / 1000,
    }

    weight_counts = {
        "Plan Time": np.array(plan) / 1000,
        "Execution Time": np.array(execs) / 1000,
    }

    width = 0.3
    fig, ax = plt.subplots()

    x = np.array([i for i in range(8)])
    labels = [
        "TrueCard",
        "DeepDB",
        "DHist",
        "FactorJoin",
        "FLAT",
        "NeuroCard",
        "WJSample",
        "Postgres",
    ]

    idx = 0
    bottom = np.zeros(8)
    for boolean, weight_count in weight_counts_10.items():
        # print(boolean, weight_count)
        idx += 1
        p = ax.bar(
            x - 0.5 * width,
            weight_count,
            width,
            label=boolean,
            bottom=bottom,
            alpha=0.3 + idx * 0.06,
        )
        bottom += weight_count
    bottom = np.zeros(8)
    for boolean, weight_count in weight_counts.items():
        # print(boolean, weight_count)
        p = ax.bar(x + 0.5 * width, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    # ax.set_title("Number of penguins with above average body mass")
    ax.legend(loc="center right")
    plt.yscale("log")
    plt.ylim([0.05, 5000])
    plt.xticks(x, labels, rotation=70)
    # add_value_labels(ax)
    w_10_1 = weight_counts_10["Plan Time (simple query)"]
    w_10_2 = w_10_1 + weight_counts_10["Execution Time (simple query)"]
    w_1 = weight_counts["Plan Time"]
    w_2 = w_1 + weight_counts["Execution Time"]
    for i in range(8):
        ax.annotate(
            "{:.2f}".format(w_10_1[i]),  # Use `label` as label
            (i - 0.5 * width, w_10_1[i]),  # Place label at end of the bar
            ha="center",  # Horizontally center label
        )
        ax.annotate(
            "{:.1f}".format(w_10_2[i]),  # Use `label` as label
            (i - 0.5 * width, w_10_2[i]),  # Place label at end of the bar
            ha="center",  # Horizontally center label
        )
        ax.annotate(
            "{:.2f}".format(w_1[i]),  # Use `label` as label
            (i + 0.5 * width, w_1[i]),  # Place label at end of the bar
            ha="center",  # Horizontally center label
        )
        ax.annotate(
            "{:.1f}".format(w_2[i]),  # Use `label` as label
            (i + 0.5 * width, w_2[i]),  # Place label at end of the bar
            ha="center",  # Horizontally center label
        )

    plt.ylabel("Total time (s)")
    # plt.xlabel("Method")
    plt.tight_layout()
    plt.show()


def add_value_labels(ax, spacing=5, bottoms=None):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    i = 0
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        if y_value < 80:
            continue
        else:
            y_value = (
                rect.get_height() if bottoms is None else rect.get_height() + bottoms[i]
            )
        i += 1
        print("y values", y_value)
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
    # plt_end_to_end()
    plt_end_to_end_all()
