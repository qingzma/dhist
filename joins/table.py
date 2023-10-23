import pandas as pd

from joins.base_logger import logger
from joins.pdf.kde import Kde1D, Kde2D
from joins.pdf.kdepy import KdePy1D, KdePy2D, plot1d, plot2d
from joins.pdf.normalizing_flow.nflow import Nflow2D


class TableContainer:
    def __init__(self) -> None:
        self.df = None
        self.name = None
        self.size = None
        self.file_path = None
        self.pdfs = dict()
        self.correlations = dict()

    def fit(self, file, join_keys, exclude=None, use_2d_model=True, args=None) -> None:
        df = pd.read_csv(file, sep=',')
        self.size = df.shape[0]
        self.file_path = file
        self.name = file.split('/')[-1].split('.')[0]

        # currently only support at most 2 join keys in a table
        assert len(join_keys) <= 2

        if len(join_keys) == 1 or (not use_2d_model):
            for join_key in join_keys:
                df_col = df[join_key].fillna(-1)  # replace NULL with -1 !
                column = Column()
                column.fit(df_col, self.name, args=args)
                self.pdfs[join_key] = column
        else:
            # replace NULL with -1 !
            df[list(join_keys)] = df[list(join_keys)].fillna(-1)
            columns = Column2d()
            columns.fit(df[list(join_keys)], self.name, args=args)
            self.pdfs[','.join(list(join_keys))] = columns


class Column:
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.pdf = None

    def fit(self, df_column, table_name, method="fft", args=None) -> None:
        """
        methods: ["fft", "kde","nflow"]
        """
        logger.debug("fit the 1d pdf...")
        self.name = df_column.head()
        if method == "kde":
            kde = Kde1D()
            kde.fit(df_column.to_numpy().reshape(-1, 1),
                    header=self.name, table=table_name)
            self.pdf = kde
            if args.plot:
                kde.plot()
        elif method == "nflow":
            logger.warning("1d nflow is not implemented yet")
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

    def fit(self, df_columns, table_name, method="fft",  args=None) -> None:
        """
        methods: ["fft", "kde","nflow"]
        """
        # use float32 for pytorch compatibility
        df_columns = df_columns.astype('float32')
        logger.debug("fit the 2d pdf...")
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
        else:
            kde = KdePy2D()
            kde.fit(df_columns.to_numpy(),
                    grid_size=args.grid, kernel=args.kernel)
            self.pdf = kde
            if args.plot:
                plot2d(kde)

    # def plot(self) -> None:
    #     logger.info("plot the pdf...")


if __name__ == '__main__':
    table = TableContainer()
    table.fit(file='data/pm25_100.csv', join_keys=['PRES'], exclude=['cbwd'])
