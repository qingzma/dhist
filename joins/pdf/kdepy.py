# https://github.com/tommyod/KDEpy
# Fast Kernel Density Estimation in Python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from matplotlib import ticker
from scipy.interpolate import (BarycentricInterpolator, CubicSpline,
                               KroghInterpolator, PchipInterpolator,
                               RegularGridInterpolator)

# kernel = "box"  # "box",gaussian


class KdePy1D:
    def __init__(self) -> None:
        self.kde = None
        self.min = None
        self.max = None

    def fit(self, x, grid_size=2**10, kernel="box"):
        x, p = FFTKDE(bw="ISJ", kernel=kernel).fit(x)(grid_size)  # "ISJ",500
        sums = np.sum(p[:-2])*(x[2]-x[1])  # [:-2])*(x[2]-x[1])  #np.sum(
        # print("sums is: ", sums)
        p = p/sums
        self.kde = PchipInterpolator(x, p)  # PchipInterpolator
        self.min = np.min(x)
        self.max = np.max(x)

    def predict(self, x):
        return self.kde(x)


class KdePy2D:
    def __init__(self) -> None:
        self.kde = None
        self.min = None
        self.max = None

    def fit(self, x, grid_size=2**10, kernel="box"):
        # print("train")
        x, p = FFTKDE(bw=10, kernel=kernel).fit(
            x)((grid_size, grid_size))
        xx, yy = np.unique(x[:, 0]), np.unique(x[:, 1])
        self.min = [xx[0], yy[0]]
        self.max = [xx[-1], yy[-1]]
        pp = p.reshape(grid_size, grid_size).T
        self.kde = RegularGridInterpolator((xx, yy), pp)
        # print("done")

    def predict(self, x):
        # only support 1 point at this moment
        return self.predict_grid([x[0]], [x[1]])

    def predict_grid(self, x_grid, y_grid):
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        return self.kde((X, Y), method="slinear")


def plot1d(kde: KdePy1D):
    x = np.linspace(kde.min, kde.max,  2**8)
    p = kde.predict(x)
    plt.plot(x, p, c='r')
    plt.show()


def plot2d(kde: KdePy2D):
    fig = plt.figure()
    ax = fig.gca()
    N = 4  # Number of contours
    xx = np.linspace(kde.min[0], kde.max[0],  2**8)
    yy = np.linspace(kde.min[1], kde.max[1],  2**8)
    p = kde.predict_grid(xx, yy)
    cfset = ax.contourf(xx, yy, p, N, cmap="Blues",
                        locator=ticker.LogLocator())
    cset = ax.contour(xx, yy, p, N, linewidths=0.8,
                      colors="k", locator=ticker.LogLocator())
    # ax.clabel(cset, inline=1, fontsize=10)
    cbar = fig.colorbar(cfset)
    plt.show()


if __name__ == "__main__":
    # n = 15
    # data = np.concatenate((np.random.randn(n), np.random.randn(n) + 10))
    # kde1d = KdePy1D()
    # kde1d.fit(data)
    # print(min(data), max(data))
    # x = np.linspace(6, 8, 2**8)
    # p = kde1d.predict(x)
    # plt.plot(x, p, c='r')
    # x, p = FFTKDE(bw="ISJ").fit(data)(2**8)
    # plt.scatter(x, p)
    # plt.show()
    # plot1d(kde1d)

    # plt.title("Fast 2D computations\nusing binning and FFT", fontsize=12)
    n = 16
    def gen_random(n): return np.random.randn(n).reshape(-1, 1)
    data1 = np.concatenate((gen_random(n), gen_random(n)), axis=1)
    data2 = np.concatenate((gen_random(n) + 1, gen_random(n) + 4), axis=1)
    data = np.concatenate((data1, data2))
    # print(data)

    kde2d = KdePy2D()
    kde2d.fit(data)
    xxx, p = FFTKDE(bw=1).fit(data)((2**8, 2**8))
    xx, yy = np.unique(xxx[:, 0]), np.unique(xxx[:, 1])
    ps = kde2d.predict_grid(xx, yy)
    print(ps)
    kde2d.predict([1, 2])
    plot2d(kde2d)
