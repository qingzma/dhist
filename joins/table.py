import pandas as pd
from joins.base_logger import logger
from joins.pdf import Kde


class Table:
    def __init__(self) -> None:
        self.df = None
        self.name = None
        self.size = None
        self.file_path = None
        self.pdfs = dict()
        self.correlations = dict()

    def fit(self, file, join_keys, exclude=None) -> None:
        df = pd.read_csv(file, sep=',')
        self.size = df.shape[0]
        self.file_path = file
        self.name = file.split('/')[-1].split('.')[0]

        # currently only support at most 2 join keys in a table
        assert len(join_keys) <= 2

        for join_key in join_keys:
            df_col = df[join_key]
            column = Column()
            column.fit(df_col)
            self.pdfs[join_key] = column


class Column:
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.pdf = None

    def fit(self, df_column) -> None:
        logger.info("fit the pdf...")
        kde = Kde()
        kde.fit(df_column.to_numpy().reshape(-1, 1))
        self.pdf = kde
        kde.plot()
    # def plot(self) -> None:
    #     logger.info("plot the pdf...")


if __name__ == '__main__':
    table = Table()
    table.fit(file='data/pm25_100.csv', join_keys=['PRES'], exclude=['cbwd'])
