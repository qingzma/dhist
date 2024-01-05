import matplotlib.pyplot as plt


def plot_line(x):
    plt.plot(x)
    plt.show()


def plot_vec_sel_array(sels, pred):
    for sel in sels:
        plt.plot(sels[sel])
    plt.plot(pred)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
