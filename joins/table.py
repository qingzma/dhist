import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from joins.base_logger import logger
from joins.pdf.fast_kde import FastKde1D, FastKde2D
from joins.pdf.kde import Kde1D, Kde2D
from joins.pdf.kdepy import KdePy1D, KdePy2D, plot1d
from joins.pdf.normalizing_flow.nflow import Nflow2D


class TableContainer:
    def __init__(self) -> None:
        # self.df = None
        self.name = None
        self.size = None
        self.file_path = None
        self.pdfs = dict()
        self.cdfs = dict()
        self.correlations = dict()
        self.correlations_cdf = dict()
        self.efficients = dict()
        self.use_cdf = False

    def fit(self, file, join_keys, relevant_keys, exclude=None, use_2d_model=True, args=None) -> None:
        df = pd.read_csv(file, sep=',')
        self.size = df.shape[0]
        self.file_path = file
        self.name = file.split('/')[-1].split('.')[0]

        if args.cdf:
            self.use_cdf = args.cdf

        # currently only support at most 1 join keys in a table
        # print("join_keys", join_keys)
        # assert len(join_keys) == 1
        # if len(join_keys) == 1 or (not use_2d_model):
        for join_key in join_keys[self.name]:
            df_col = df[join_key].fillna(-1)  # replace NULL with -1 !
            column = Column()
            column.fit(df_col, self.name, args=args)
            self.pdfs[join_key] = column

        for relev_key in relevant_keys[self.name]:
            logger.info("col is %s", relev_key)
            df_col = df[relev_key].fillna(-1)  # replace NULL with -1 !
            column = Column()
            column.fit(df_col, self.name, args=args)
            if args.cdf:
                self.cdfs[relev_key] = column
            else:
                self.pdfs[relev_key] = column
        # else:
        # replace NULL with -1 !
        # df[list(join_keys)] = df[list(join_keys)].fillna(-1)
        # columns = Column2d()
        # columns.fit(df[list(join_keys)], self.name, args=args)
        # self.pdfs[','.join(list(join_keys))] = columns

        # print(join_keys)
        # print(relevant_keys)
        # for t in join_keys:
        print("name is ", self.name)
        t = self.name
        for join_key in join_keys[t]:
            if t in relevant_keys:
                for relevant_key in relevant_keys[t]:
                    # print("relevant_key", relevant_key)
                    if relevant_key != join_key:
                        columns = Column2d()
                        # print(df)
                        # print(df.columns)
                        # print("min of join key is ", df[join_key].min())
                        # print("max of join key is ", df[join_key].max())
                        # print("min of relevant key is ",
                        #       df[relevant_key].min())
                        # print("max of relevant key is ",
                        #       df[relevant_key].max())
                        # print([join_key, relevant_key])
                        d = df[[join_key, relevant_key]].fillna(-1)
                        # print(d)
                        # exit()
                        columns.fit(d,
                                    self.name, args=args)
                        # if t not in self.correlations:
                        #     self.correlations[t] = dict()
                        if args.cdf:
                            if join_key not in self.correlations_cdf:
                                self.correlations_cdf[join_key] = dict()
                            self.correlations_cdf[join_key][relevant_key] = columns
                        else:
                            if join_key not in self.correlations:
                                self.correlations[join_key] = dict()
                            self.correlations[join_key][relevant_key] = columns
        # exit()


class Column:
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.pdf = None
        self.min = None
        self.max = None

    def fit(self, df_column, table_name, method="fast", args=None) -> None:
        """
        methods: ["fft", "kde","nflow"]
        """
        x = df_column.to_numpy().reshape(-1, 1)
        logger.debug("fit the 1d pdf...")
        self.min = np.min(x)
        self.max = np.max(x)
        self.name = df_column.head()
        self.size = len(x)
        if method == "kde":
            kde = Kde1D()
            kde.fit(x,
                    header=self.name, table=table_name)
            self.pdf = kde
            if args.plot:
                kde.plot()
        elif method == "nflow":
            logger.warning("1d nflow is not implemented yet")
        elif method == "fast":
            kde = FastKde1D(grid_size=args.grid, cumulative=args.cdf)
            kde.fit(df_column.to_numpy().reshape(-1, 1))
            self.pdf = kde
            if args.plot:
                plot1d(kde)
        else:
            kde = KdePy1D()
            kde.fit(df_column.to_numpy().reshape(-1, 1),
                    grid_size=args.grid, kernel=args.kernel)
            self.pdf = kde
            if args.plot:
                plot1d(kde)
        # kde.plot()
    # def plot(self) -> None:
    #     logger.info("plot the pdf...")


class Column2d:
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.pdf = None
        self.min = None
        self.max = None

    def fit(self, df_columns, table_name, method="fast",  args=None, use_coefficient=False) -> None:
        """
        methods: ["fft", "kde","nflow"]
        """
        self.size = df_columns.shape[0]
        # use float32 for pytorch compatibility
        df_columns = df_columns.astype('float32')
        logger.debug("fit the 2d pdf...")
        # logger.info("mins: %s", df_columns.min())
        self.min = df_columns.min().values
        self.max = df_columns.max().values
        # exit()
        self.name = df_columns.head()
        if method == 'nflow':
            flow = Nflow2D(max_iter=100, show_iter=200, enable_cuda=False)
            flow.fit(df_columns.to_numpy())
            self.pdf = flow
            if args.plot:
                flow.plot()
        elif method == 'kde':
            kde = Kde2D()
            kde.fit(df_columns.to_numpy(),  # .reshape(-1, 2),
                    header=self.name, table=table_name)
            self.pdf = kde
            if args.plot:
                kde.plot()
        elif method == "fast":
            kde = FastKde2D(args.grid, args.grid, cumulative=args.cdf)
            kde.fit(df_columns.to_numpy())
            self.pdf = kde
            if args.plot:
                plot2d(kde)
        else:
            if use_coefficient:
                pass
                # kde = KdePy2DEfficient()
                # kde.fit(df_columns.to_numpy(),
                #         grid_size=args.grid, kernel=args.kernel)
                # self.pdf = kde
            else:
                kde = KdePy2D()
                kde.fit(df_columns.to_numpy(),
                        grid_size=args.grid, kernel=args.kernel)
                self.pdf = kde
            if args.plot:
                plot2d(kde)

    # def plot(self) -> None:
    #     logger.info("plot the pdf...")


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


if __name__ == '__main__':
    table = TableContainer()
    # table.fit(file='data/pm25_100.csv', join_keys=['PRES'], exclude=['cbwd'])
