import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from joins.base_logger import logger
from joins.cnts.cnts import CumulativeDistinctCounter
from joins.histograms.histograms import UpperBoundHistogramTopK
from joins.histograms.non_key_histogram import (
    NonKeyCumulativeHistogram,
    NonKeyTopKHistogram,
)
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

    def fit(self, file, join_keys, relevant_keys, bin_info, args=None) -> None:
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
            df_col = df[[join_key]]  # .fillna(-1)  # replace NULL with -1 !
            column = KeyColumnTopK()
            # print("bin_info", bin_info)
            # print("join key", join_key)
            if self.name in bin_info and join_key in bin_info[self.name]:
                bins = np.linspace(
                    bin_info[self.name][join_key][0],
                    bin_info[self.name][join_key][1],
                    args.grid,
                )
                column.fit(df_col, bins=bins, args=args)
                self.key_hist[join_key] = column

            # counter = CumulativeDistinctCounter()
            # counter.fit(df[join_key].fillna(-1))
            # self.counters[join_key] = counter

        for relev_key in relevant_keys[self.name]:
            # logger.info("col is %s", relev_key)
            df_col = df[[relev_key]]  # .fillna(-1)  # replace NULL with -1 !
            column = NonKeyColumnTopK()
            column.fit(df_col, args=args)
            self.non_key_hist[relev_key] = column
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

    def fit(self, df_column, bins, args=None) -> None:
        kde = UpperBoundHistogramTopK(args.topk)
        kde.fit(df_column, df_column.columns, bins=bins)
        self.pdf = kde


class NonKeyColumnTopK:
    def __init__(self) -> None:
        self.name = None
        self.pdf = None
        self.min = None
        self.max = None

    def fit(self, df_column, args=None) -> None:
        if args.cdf:
            kde = NonKeyCumulativeHistogram(n_bins=args.grid)

        else:
            kde = NonKeyTopKHistogram(
                n_bins=args.grid, n_top_k=args.topk, n_categorical=200
            )
        kde.fit(df_column, headers=df_column.columns)
        self.pdf = kde
