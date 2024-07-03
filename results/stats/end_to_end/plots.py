import matplotlib.pyplot as plt
import numpy as np

from joins.tools import read_from_csv_all


def plt_all():
    truth = read_from_csv_all("results/stats/end_to_end/truth_10s.csv")
    deepdb = read_from_csv_all("results/stats/end_to_end/deepdb_10s.csv")
    dhist = read_from_csv_all("results/stats/end_to_end/dhist_10s.csv")
    factorjoin = read_from_csv_all("results/stats/end_to_end/factorjoin_10s.csv")
    flat = read_from_csv_all("results/stats/end_to_end/flat_10s.csv")
    neurocard = read_from_csv_all("results/stats/end_to_end/neurocard_10s.csv")
    wjsample = read_from_csv_all("results/stats/end_to_end/wjsample_10s.csv")
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

    data = [truth, deepdb, dhist, factorjoin, flat, neurocard, wjsample, postgres]
    # print("idx is ", idx)
    # "plan-time"
    exec = []
    plan = []
    for d in data:
        exec.append(np.sum(d["execution-time"].values[idx]))
        plan.append(np.sum(d["plan-time"].values[idx]))

    print(exec)
    print(plan)
    # plt.show()


if __name__ == "__main__":
    plt_all()
