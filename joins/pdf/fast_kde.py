import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.interpolate import (BarycentricInterpolator, CubicSpline,
                               KroghInterpolator, PchipInterpolator,
                               RegularGridInterpolator, interp2d)

from joins.domain import Domain


def get_linspace_centered(low: float, high: float, sz: int):
    """put grid point in bin center, avoid low bound mismatch

    Args:
        low (float): lower bound
        high (float): upper bound
        sz (int): size of grid

    Returns:
        grid, width: grid and width
    """
    grid_width_x = (high-low)/sz
    x_low_in_grid = low - 0.5*grid_width_x
    x_high_in_grid = high + 0.5*grid_width_x
    grid_x = np.linspace(x_low_in_grid, x_high_in_grid, sz)
    return grid_x, grid_width_x


class FastKde1D:
    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size
        # self.grid_width = None
        self.min = None
        self.max = None
        # self.grid = None
        self.size = None
        self.kde = None
        self.background_noise = None

    def fit(self, x) -> None:
        if isinstance(x, pd.DataFrame):
            self.fit_pd(x)
            return
        if isinstance(x, np.ndarray):
            self.fit_numpy(x)
            return

        if isinstance(x, list):
            self.fit_list(x)
            return
        print("other data format is not supported yet.")
        return

    def fit_pd(self, df: pd.DataFrame):
        """_summary_

        Args:
            x (pd.DataFrame): first column be x, second column be y
        """
        self.size = df.size
        self.min = df.min().to_numpy()[0]
        self.max = df.max().to_numpy()[0]
        column = list(df.columns)[0]

        grid_x, _ = get_linspace_centered(
            self.min, self.max, self.grid_size)
        df[column] = pd.cut(
            df[column], bins=grid_x, labels=grid_x[:-1])  # , labels=self.grid_x[:-1]

        counts = df.groupby([column],
                            observed=False).size().to_numpy()  # [['x', 'y']].count()  .size()

        xx, wx = np.linspace(
            self.min, self.max, self.grid_size-1, retstep=True)

        ps = np.divide(counts, self.size*wx)

        self.background_noise = 1/self.size/wx

        self.kde = PchipInterpolator(xx, ps)

    def fit_numpy(self, x: np.ndarray):
        df = pd.DataFrame(x, columns=['x'])
        self.fit_pd(df)

    def fit_list(self, x: list):
        dat = np.array(x)
        self.fit_numpy(dat)

    def predict(self, x):
        res = self.kde(x)
        # res[np.logical_and(res < self.background_noise, res > 0)] = 0
        res[res < self.background_noise] = 0
        return res


class FastKde2D:
    def __init__(self, grid_size_x, grid_size_y, cumulative=False) -> None:
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.background_noise = 0
        self.min = None
        self.max = None
        self.size = None
        self.kde = None
        self.cumulative = cumulative

    def predict(self, x):
        # only support 1 point at this moment
        return self.predict_grid([x[0]], [x[1]])

    def predict_grid_with_y_range(self, x_grid, domain: Domain, b_plot=True):
        assert (self.cumulative)
        l, h = domain.min, domain.max
        if domain.left:
            l -= 0.05
        if not domain.right:
            h -= 0.05
        p_l = self.predict_grid(x_grid, [l])
        p_h = self.predict_grid(x_grid, [h])
        if b_plot:
            plot2d(self)
        print("----low is %s",p_l)
        print("----high is %s",p_h)
        return np.subtract(p_h, p_l).reshape(1, -1)[0]

    def predict_grid(self, x_grid, y_grid):
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        res = self.kde((X, Y))
        res[res < self.background_noise] = 0
        return res

    def fit(self, x) -> None:
        if isinstance(x, pd.DataFrame):
            self.fit_pd(x)
            return
        if isinstance(x, np.ndarray):
            self.fit_numpy(x)
            return

        if isinstance(x, list):
            self.fit_list(x)
            return
        print("other data format is not supported yet.")
        return

    def fit_pd(self, df: pd.DataFrame):
        """_summary_

        Args:
            x (pd.DataFrame): first column be x, second column be y
        """
        self.size = df.shape[0]
        self.min = df.min().to_numpy()
        self.max = df.max().to_numpy()
        columns = list(df.columns)

        grid_x, _ = get_linspace_centered(
            self.min[0], self.max[0], self.grid_size_x)
        grid_y, _ = get_linspace_centered(
            self.min[1], self.max[1], self.grid_size_y)
        df[columns[0]] = pd.cut(
            df[columns[0]], bins=grid_x, labels=grid_x[:-1])  # , labels=self.grid_x[:-1]
        df[columns[1]] = pd.cut(
            df[columns[1]], bins=grid_y, labels=grid_y[:-1])  # , labels=grid_y[:-1]

        counts = df.groupby(columns,
                            observed=False).size().unstack().to_numpy()  # [['x', 'y']].count()  .size()
        if self.cumulative:
            counts = np.cumsum(counts, axis=0)
        # print("counts is ", counts)
        # print("sum of count is ", np.sum(counts))
        # print("table is ", self.size)
        xx, wx = np.linspace(
            self.min[0], self.max[0], self.grid_size_x-1, retstep=True)
        yy, wy = np.linspace(self.min[1], self.max[1],
                             self.grid_size_y-1, retstep=True)
        ps = np.divide(counts, self.size*wx*wy)
        print("sum is ", np.sum(ps[-1])*wx*wy)
        self.background_noise = 1.0/self.size/wx/wy

        self.kde = RegularGridInterpolator((xx, yy), ps)

    def fit_numpy(self, x: np.ndarray):
        df = pd.DataFrame(x, columns=['x', 'y'])
        self.fit_pd(df)

    def fit_list(self, x: list):
        dat = np.array(x)
        self.fit_numpy(dat)


def plot1d(kde):
    x = np.linspace(kde.min, kde.max,  2**8)
    p = kde.predict(x)
    plt.plot(x, p, c='r')
    plt.show()


def plot2d(kde):
    fig = plt.figure()
    ax = fig.gca()
    N = 4  # Number of contours
    xx = np.linspace(kde.min[0], kde.max[0],  2**10)
    yy = np.linspace(kde.min[1], kde.max[1],  2**10)
    p = kde.predict_grid(xx, yy)
    cfset = ax.contourf(xx, yy, p, N, cmap="Blues",
                        locator=ticker.LogLocator())
    cset = ax.contour(xx, yy, p, N, linewidths=0.8,
                      colors="k", locator=ticker.LogLocator())
    # ax.clabel(cset, inline=1, fontsize=10)
    cbar = fig.colorbar(cfset)
    plt.show()


if __name__ == "__main__":
    # n = 1000000
    # data = np.concatenate((np.random.randn(n), np.random.randn(n) + 10))
    # kde1d = FastKde1D(2**12)
    # kde1d.fit(data)
    # print(min(data), max(data))
    # # x = np.linspace(-5, 15, 2**8)
    # # p = kde1d.predict(x)
    # # plt.plot(x, p, c='r')
    # # plt.show()
    # plot1d(kde1d)

    # n = 300000
    # def gen_random(n): return np.random.randn(n).reshape(-1, 1)
    # data1 = np.concatenate((gen_random(n), gen_random(n)), axis=1)
    # data2 = np.concatenate((gen_random(n) + 1, gen_random(n) + 4), axis=1)
    # data = np.concatenate((data1, data2))
    # # print(data)

    # kde2d = FastKde2D(5, 10, cumulative=True)
    # kde2d.fit(data)

    # xx, yy = np.linspace(-4, 4, 10), np.linspace(-4, 4, 10)
    # ps = kde2d.predict_grid(xx, yy)
    # print(ps)
    # domain = Domain(0.5, 1, True, True)
    # pss = kde2d.predict_grid_with_y_range(xx, domain)
    # print("pss is ", pss)
    # kde2d.predict([1, 2])
    # plot2d(kde2d)

    data = [[1, 2],
            [1, 3],
            [2, 3],
            [4, 5]]
    kde2d = FastKde2D(3,4,cumulative=True)
    kde2d.fit(data)
    # domain = Domain(0.8, 4.2, True, True)
    # kde2d.predict_grid_with_y_range(np.linspace(1.2,3.8,10),domain,b_plot=True)
    plot2d(kde2d)
