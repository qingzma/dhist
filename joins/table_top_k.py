import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from joins.base_logger import logger
from joins.cnts.cnts import CumulativeDistinctCounter
from joins.pdf.fast_kde import FastKde1D, FastKde2D
from joins.pdf.kde import Kde1D, Kde2D
from joins.pdf.kdepy import KdePy1D, KdePy2D, plot1d
from joins.pdf.normalizing_flow.nflow import Nflow2D


class TableContainerTopK:
    def __init__(self) -> None:
        # self.df = None
        self.name = None
        self.size = None
        self.file_path = None
        # self.counters = dict()
        self.key_hist = {}
        self.non_key_hist = {}

    def fit(self, file, join_keys, relevant_keys, args=None) -> None:
        df = pd.read_csv(file, sep=",")
        self.size = df.shape[0]
        self.file_path = file
        self.name = file.split("/")[-1].split(".")[0]

        # if args.cdf:
        #     self.use_cdf = args.cdf

        # currently only support at most 1 join keys in a table
        # print("join_keys", join_keys)
        # assert len(join_keys) == 1
        # if len(join_keys) == 1 or (not use_2d_model):

        for join_key in join_keys[self.name]:
            df[join_key] = df[join_key]  # .fillna(-1)
            df_col = df[join_key]  # .fillna(-1)  # replace NULL with -1 !
            column = Column()
            column.fit(df_col, self.name, args=args)
            self.pdfs[join_key] = column

            # counter = CumulativeDistinctCounter()
            # counter.fit(df[join_key].fillna(-1))
            # self.counters[join_key] = counter

        for relev_key in relevant_keys[self.name]:
            # logger.info("col is %s", relev_key)
            df_col = df[relev_key]  # .fillna(-1)  # replace NULL with -1 !
            column = Column()
            column.fit(df_col, self.name, args=args)
            # if args.cdf:
            #     self.cdfs[relev_key] = column
            # else:
            #     self.pdfs[relev_key] = column


class KeyColumnTopK:
    def __init__(self) -> None:
        self.name = None
        self.pdf = None
        self.min = None
        self.max = None

    def fit(self, df_column, table_name, args=None) -> None:
        kde = FastKde1D(grid_size=args.grid, cumulative=False)
        kde.fit(df_column.to_numpy().reshape(-1, 1))
        self.pdf = kde


class NonKeyColumnTopK:
    def __init__(self) -> None:
        self.name = None
        self.pdf = None
        self.min = None
        self.max = None

    def fit(self, df_column, table_name, args=None) -> None:
        kde = FastKde1D(grid_size=args.grid, cumulative=False)
        kde.fit(df_column.to_numpy().reshape(-1, 1))
        self.pdf = kde
