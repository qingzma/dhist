import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.interpolate import (
    BarycentricInterpolator,
    CubicSpline,
    KroghInterpolator,
    PchipInterpolator,
    RegularGridInterpolator,
    interp2d,
    Akima1DInterpolator,
)

# from numba import njit

from joins.domain import Domain


# @njit
# def interp_nb(x_vals, x, y):
#     return np.interp(xvals, x, y)


def get_linspace_centered(low: float, high: float, sz: int):
    """put grid point in bin center, avoid low bound mismatch

    Args:
        low (float): lower bound
        high (float): upper bound
        sz (int): size of grid

    Returns:
        grid, width: grid and width
    """
    grid_width_x = (high - low) / sz
    x_low_in_grid = low - 0.5 * grid_width_x
    x_high_in_grid = high + 0.5 * grid_width_x
    grid_x = np.linspace(x_low_in_grid, x_high_in_grid, sz)
    return grid_x, grid_width_x


class FastKde1D:
    def __init__(self, grid_size, cumulative=True, is_categorical=False) -> None:
        self.grid_size = grid_size
        # self.grid_width = None
        self.min = None
        self.max = None
        # self.grid = None
        self.size = None
        self.kde = None
        self.background_noise = None
        self.cumulative = cumulative
        self.is_categorical = is_categorical
        if is_categorical:
            # when categorical is on, must use cumulative mode, otherwise wrong density as no 0 values.
            assert cumulative

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

        if self.is_categorical:
            unique_x = np.unique(df[column])
            unique_x.sort()
            grid_x = unique_x - 0.5
            grid_x = np.append(grid_x, [unique_x[-1] + 0.5])
        else:
            width_x = (self.max - self.min) / (self.grid_size - 1)
            grid_x, _ = get_linspace_centered(
                self.min - 0.5 * width_x, self.max + 0.5 * width_x, self.grid_size
            )
        df[column] = pd.cut(
            df[column], bins=grid_x, labels=grid_x[:-1]
        )  # , labels=self.grid_x[:-1]

        counts = (
            df.groupby([column], observed=False).size().to_numpy()
        )  # [['x', 'y']].count()  .size()
        # print("counts was ", counts)
        if self.cumulative:
            counts = np.cumsum(counts, axis=0)
            # print("counts is ", counts)
            # exit()

        if self.is_categorical:
            xx = unique_x
            ps = np.divide(counts, 1.0 * self.size)
            self.background_noise = 0.5 / self.size
        else:
            xx, wx = np.linspace(self.min, self.max,
                                 self.grid_size - 1, retstep=True)
            if self.cumulative:
                ps = np.divide(counts, self.size * wx)
                self.background_noise = 0.5 / self.size / wx
            else:
                ps = np.divide(counts, self.size * wx)
                self.background_noise = 0.5 / self.size / wx
        # print("ps is ", ps)
        # print("background noise is ", self.background_noise)

        # Akima1DInterpolator, PchipInterpolator, CubicSpline
        self.kde = PchipInterpolator(xx, ps)

    def fit_numpy(self, x: np.ndarray):
        df = pd.DataFrame(x, columns=["x"])
        self.fit_pd(df)

    def fit_list(self, x: list):
        dat = np.array(x)
        self.fit_numpy(dat)

    def predict(self, x):
        res = self.kde(x)
        # res[np.logical_and(res < self.background_noise, res > 0)] = 0
        res[res < self.background_noise] = 0
        return res

    def predict_domain(self, domain: Domain):
        assert self.cumulative
        l = domain.min
        h = domain.max

        # support  (l, h) seamlessly, [1,1] is treated as (0,2)
        if domain.left:
            l -= 1
        if domain.right:
            h += 1
        try:
            ps = self.predict(np.array([l, h]))
        except Exception:
            print("error, out of bound, pls restore lower and uppper bound.")
            exit()
        return ps[1] - ps[0]


class FastKde2D:
    def __init__(
        self, grid_size_x, grid_size_y, cumulative=False, y_is_categorical=False
    ) -> None:
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.background_noise = 0
        self.min = None
        self.max = None
        self.size = None
        self.kde = None
        self.cumulative = cumulative
        self.y_is_categorical = y_is_categorical
        self.y_categorical_half_width = 0.5

    def predict(self, x):
        # only support 1 point at this moment
        return self.predict_grid([x[0]], [x[1]])

    def predict_grid_with_y_range(self, x_grid, domain: Domain, b_plot=False):
        print("domain is  \n", domain)
        assert self.cumulative
        l, h = domain.min, domain.max
        # width = x_grid[1]-x_grid[0]
        # if domain.is_categorical:
        #     if domain.left:

        # else:
        # if l == h:
        #     h += 0.006

        # if domain.left:
        #     l -= 0.005
        # if not domain.right:
        #     h -= 0.005
        if b_plot:
            plot2d(self)

        # support  (l, h) seamlessly, [1,1] is treated as (0,2)

        # if domain.max > self.max[1] and domain.min < self.min[1]:
        #     ps = self.predict_grid(x_grid, np.array([self.max]))
        # elif domain.max == self.max[1] and domain.min < self.min[1]:

        # if domain.left:
        #     l = max(self.min[1], l - 0.5)
        # else:
        #     l += 0.5
        # if domain.right:
        #     h = min(self.max[1], h + 0.50)
        # else:
        #     h -= 0.5
        print("min is %s", self.min)
        print("max is %s", self.max)

        if domain.left:
            l -= 0.5
        else:
            l += 0.5
            # pass
        if domain.right:
            # pass
            h += 0.5
        else:
            h -= 0.5

        try:
            ps = self.predict_grid(x_grid, np.array([l, h]))
        except ValueError:
            try:
                hh = h - 0.5
                ps = self.predict_grid(x_grid, np.array([l, self.max[1]]))
                print(
                    "try to  restore  uppper bound to fix issue. %s",
                    Domain(l, hh, False, False),
                )
            except ValueError:
                try:
                    ll = l + 0.5
                    ps = self.predict_grid(x_grid, np.array([h]))
                    print(
                        "try to  restore  lower bound to fix issue. %s",
                        Domain(ll, h, False, False),
                    )
                except ValueError:
                    # hh = h - 0.5
                    # ll = l + 0.5

                    print("hh %s", hh)
                    print("ll %s", ll)
                    print("min of grid is %s", x_grid[0])
                    print("max of grid is %s", x_grid[-1])
                    # TODO need to optimize this part, avoid multiple try.
                    # ps = self.predict_grid(x_grid, np.array([ll, hh]))
                    ps = self.predict_grid(x_grid, np.array([h - 0.5]))
                    print(
                        "try to  restore  both bounds to fix issue, this should not happen. %s",
                        Domain(ll, hh, False, False),
                    )
            # exit()
        # if len(ps) == 1:
        #     return ps
        # return ps[1] - ps[0]
        if len(ps) == 1:
            return ps[0]
        return ps[1] - ps[0]

    def predict_grid(self, x_grid, y_grid, b_plot=False):
        if b_plot:
            plot2d(self)
        X, Y = np.meshgrid(x_grid, y_grid)  # ,indexing='ij'

        X, Y = np.meshgrid(x_grid, y_grid)  # ,indexing='ij'

        res = self.kde((X, Y))
        res[res < self.background_noise] = 0
        # print("predicted grid mesh is \n",res)
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

        width_x = (self.max[0] - self.min[0]) / (self.grid_size_x - 1)
        width_y = (self.max[1] - self.min[1]) / (self.grid_size_x - 1)
        grid_x, _ = get_linspace_centered(
            self.min[0] - 0.5 * width_x,
            self.max[0] + 0.5 * width_x,
            self.grid_size_x + 1,
        )

        if self.y_is_categorical:
            unique_y = np.unique(df[columns[1]])
            unique_y.sort()
            # print("unique y is ", unique_y)
            grid_y = unique_y - 0.5 * width_y
            grid_y = np.append(grid_y, [unique_y[-1] + 0.5 * width_y])
        else:
            grid_y, _ = get_linspace_centered(
                self.min[1] - 0.5 * width_y,
                self.max[1] + 0.5 * width_y,
                self.grid_size_y + 1,
            )
        # print("unique y is ", unique_y)
        # print("widthx was ", width_x)
        # print("original grid_x is ", grid_x)
        # print("original grid_y is ", grid_y)
        df[columns[0]] = pd.cut(
            df[columns[0]],
            bins=grid_x,
        )  # , labels=grid_x[:-1]
        # print("grid y is %s", grid_y)
        # exit()
        df[columns[1]] = pd.cut(
            df[columns[1]],
            bins=grid_y,
        )  # , labels=grid_y[:-1]

        # counts1 = df.groupby(columns,  # [columns[0],columns[1]]
        #                      observed=False).size()

        # counts1 = df.groupby(columns,  # [columns[0],columns[1]]
        #                      observed=False).size()
        # print("counts1\n",counts1)

        counts = (
            df.groupby(columns, observed=False).size().unstack().to_numpy()
        )  # [['x', 'y']].count()  .size()
        # print("first row is ", counts[0,:])
        # print("counts before is\n", counts)
        if self.cumulative:
            counts = np.cumsum(counts, axis=1)
        # print("counts is\n", counts)
        print("sum of count is ", np.sum(counts))
        print("sum of last count is ", np.sum(counts[-1, :]))
        # print("table is ", self.size)
        xx, wx = np.linspace(
            self.min[0], self.max[0], self.grid_size_x, retstep=True)
        if self.y_is_categorical:
            yy = unique_y
            # print("y grid is now ", unique_y)
            ps = np.divide(counts, self.size * wx)
            print("sum is ", np.sum(ps[:, -1]) * wx)
            # print("corresponding xx is ", xx)
            # print("corresponding yy is ", yy)
            self.background_noise = 0.5 / self.size / wx
            # print("background_noise is now ", self.background_noise)
        else:
            yy, wy = np.linspace(
                self.min[1], self.max[1], self.grid_size_y, retstep=True
            )
            ps = np.divide(counts, self.size * wx)
            # print("width x is ", wx)
            print("sum is ", np.sum(ps[:, -1]) * wx)
            self.background_noise = 0.5 / self.size / wx

        # print("x_grid is ", xx)
        # print("widthx is ", wx)
        # print("y_grid is ", yy)

        self.kde = RegularGridInterpolator((xx, yy), ps, fill_value=0)
        # print("self.kde.grid is ", self.kde.grid)

    def fit_numpy(self, x: np.ndarray):
        df = pd.DataFrame(x, columns=["x", "y"])
        self.fit_pd(df)

    def fit_list(self, x: list):
        dat = np.array(x)
        self.fit_numpy(dat)


def plot1d(kde):
    x = np.linspace(kde.min, kde.max, 2**8)
    p = kde.predict(x)
    plt.plot(x, p, c="r")
    plt.show()


def plot2d(kde, grid_size_x=2**10, grid_size_y=2**10):
    fig = plt.figure()
    ax = fig.gca()
    N = 4  # Number of contours
    xx = np.linspace(kde.min[0], kde.max[0], grid_size_x)
    yy = np.linspace(kde.min[1], kde.max[1], grid_size_y)
    p = kde.predict_grid(xx, yy, b_plot=False)
    cfset = ax.contourf(xx, yy, p, N, cmap="Blues",
                        locator=ticker.LogLocator())
    cset = ax.contour(
        xx, yy, p, N, linewidths=0.8, colors="k", locator=ticker.LogLocator()
    )
    # ax.clabel(cset, inline=1, fontsize=10)
    cbar = fig.colorbar(cfset)
    plt.show()


if __name__ == "__main__":
    # n = 10000
    # data = np.concatenate((np.random.randn(n), np.random.randn(n) + 10))
    # kde1d = FastKde1D(2**12)
    # kde1d.fit(data)
    # print(min(data), max(data))
    # # x = np.linspace(-5, 15, 2**8)
    # # p = kde1d.predict(x)
    # # plt.plot(x, p, c='r')
    # # plt.show()
    # plot1d(kde1d)

    n = 10
    data = np.array([1, 2, 3, 4, 5, 6, 7, 7, 7, 10, 10, 20])
    kde1d = FastKde1D(100, cumulative=True, is_categorical=True)
    kde1d.fit(data)
    print(min(data), max(data))
    # x = np.linspace(-5, 15, 2**8)
    # p = kde1d.predict(x)
    # plt.plot(x, p, c='r')
    # plt.show()
    plot1d(kde1d)

    # n = 3000
    # def gen_random(n): return np.random.randn(n).reshape(-1, 1)
    # data1 = np.concatenate((gen_random(n)-0.5, gen_random(n)-0.5), axis=1)
    # data2 = np.concatenate((gen_random(n) + 3.5, gen_random(n) + 7.5), axis=1)
    # data = np.concatenate((data1, data2))
    # # print(data)

    # kde2d = FastKde2D(50, 10, cumulative=True, y_is_categorical=True)
    # kde2d.fit(data)

    # xx, yy = np.linspace(1, 2, 3), np.linspace(4, 6, 4)
    # ps = kde2d.predict_grid(xx, yy)
    # print("xx", xx)
    # print("yy", yy)
    # print("ps", ps)
    # domain = Domain(0.5, 1, True, True)
    # pss = kde2d.predict_grid_with_y_range(xx, domain)
    # # print("pss is ", pss)
    # kde2d.predict([1, 2])
    # plot2d(kde2d)

    data = [
        [1, 2],
        [1, 3],
        [2, 3],
        [4, 5],
    ]
    #         [1, 2],
    #         [1, 3],
    #         [2, 3],
    #         [4, 5],
    #         [1, 2],
    #         [1, 3],
    #         [2, 3],
    #         [4, 5],
    #         [1, 2],
    #         [1, 3],
    #         [2, 3],
    #         [4, 5],
    #         [1, 2],
    #         [1, 3],
    #         [2, 3],
    #         [4, 5],
    #         [1, 2],
    #         [1, 3],
    #         [2, 3],
    #         [4, 5]]

    # kde2d = FastKde2D(2, 3, cumulative=True)
    # kde2d.fit(data)
    # domain = Domain(3, 4.2, True, True)
    # # kde2d.predict_grid_with_y_range([1,4],domain,b_plot=True)
    # plot2d(kde2d)
