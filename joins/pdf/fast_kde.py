import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.interpolate import (BarycentricInterpolator, CubicSpline,
                               KroghInterpolator, PchipInterpolator,
                               RegularGridInterpolator, interp2d)


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
        self.low = None
        self.high = None
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
        self.low = df.min().to_numpy()[0]
        self.high = df.max().to_numpy()[0]
        column = list(df.columns)[0]

        grid_x, _ = get_linspace_centered(
            self.low, self.high, self.grid_size)
        df[column] = pd.cut(
            df[column], bins=grid_x, labels=grid_x[:-1])  # , labels=self.grid_x[:-1]

        counts = df.groupby([column],
                            observed=False).size().to_numpy()  # [['x', 'y']].count()  .size()

        xx, wx = np.linspace(
            self.low, self.high, self.grid_size-1, retstep=True)

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
    def __init__(self, grid_size_x, grid_size_y) -> None:
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.background_noise = 0
        # self.grid_width_x = None
        # self.grid_width_y = None
        self.low = None
        self.high = None
        # self.grid_x = None
        self.size = None
        self.kde = None
        # self.low_bound_tolerance = 0.1

    def predict(self, x):
        # only support 1 point at this moment
        return self.predict_grid([x[0]], [x[1]])

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
        self.size = df.size
        self.low = df.min().to_numpy()
        self.high = df.max().to_numpy()
        columns = list(df.columns)

        grid_x, _ = get_linspace_centered(
            self.low[0], self.high[0], self.grid_size_x)
        grid_y, _ = get_linspace_centered(
            self.low[1], self.high[1], self.grid_size_y)
        df[columns[0]] = pd.cut(
            df[columns[0]], bins=grid_x, labels=grid_x[:-1])  # , labels=self.grid_x[:-1]
        df[columns[1]] = pd.cut(
            df[columns[1]], bins=grid_y, labels=grid_y[:-1])  # , labels=grid_y[:-1]

        counts = df.groupby(columns,
                            observed=False).size().unstack().to_numpy()  # [['x', 'y']].count()  .size()

        xx, wx = np.linspace(
            self.low[0], self.high[0], self.grid_size_x-1, retstep=True)
        yy, wy = np.linspace(self.low[1], self.high[1],
                             self.grid_size_y-1, retstep=True)
        ps = np.divide(counts, self.size*wx*wy)
        self.background_noise = 1.0/self.size/wx/wy

        self.kde = RegularGridInterpolator((xx, yy), ps)

    def fit_numpy(self, x: np.ndarray):
        df = pd.DataFrame(x, columns=['x', 'y'])
        self.fit_pd(df)
        # self.low = x.min(axis=0)
        # self.high = x.max(axis=0)
        # self.grid_x, self.grid_width_x = get_linspace_centered(
        #     self.low[0], self.high[0], self.grid_size_x)
        # grid_y, grid_width_y = get_linspace_centered(
        #     self.low[1], self.high[1], self.grid_size_y)
        # x[0, :] = np.searchsorted(self.grid_size_x, x[0, :])
        # x[1, :] = np.searchsorted(grid_y, x[1, :])

    def fit_list(self, x: list):
        dat = np.array(x)
        self.fit_numpy(dat)


def plot1d(kde):
    x = np.linspace(kde.low, kde.high,  2**8)
    p = kde.predict(x)
    plt.plot(x, p, c='r')
    plt.show()


def plot2d(kde):
    fig = plt.figure()
    ax = fig.gca()
    N = 4  # Number of contours
    xx = np.linspace(kde.low[0], kde.high[0],  2**10)
    yy = np.linspace(kde.low[1], kde.high[1],  2**10)
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

    # plt.title("Fast 2D computations\nusing binning and FFT", fontsize=12)
    n = 30000000
    def gen_random(n): return np.random.randn(n).reshape(-1, 1)
    data1 = np.concatenate((gen_random(n), gen_random(n)), axis=1)
    data2 = np.concatenate((gen_random(n) + 1, gen_random(n) + 4), axis=1)
    data = np.concatenate((data1, data2))
    # print(data)

    kde2d = FastKde2D(1000, 1000)
    kde2d.fit(data)

    xx, yy = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
    ps = kde2d.predict_grid(xx, yy)
    print(ps)
    kde2d.predict([1, 2])
    plot2d(kde2d)

# if __name__ == '__main__':
#     data = [[1, 2],
#             [1, 3],
#             [2, 3],
#             [4, 5]]
#     kde = FastKde(4, 3)
#     kde.fit(data)
