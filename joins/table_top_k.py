import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from joins.base_logger import logger
from joins.cnts.cnts import CumulativeDistinctCounter
from joins.domain import Domain
from joins.histograms.histograms import (
    JoinHistogram,
    UpperBoundHistogram,
    UpperBoundHistogramTopK,
)
from joins.histograms.non_key_histogram import (
    NonKeyCumulativeHistogram,
    NonKeyTopKHistogram,
)
from joins.pdf.fast_kde import FastKde1D, FastKde2D
from joins.pdf.kde import Kde1D, Kde2D
from joins.pdf.kdepy import KdePy1D, KdePy2D, plot1d
from joins.pdf.normalizing_flow.nflow import Nflow2D
from joins.schema_base import SchemaGraph


class TableContainerTopK:
    def __init__(self) -> None:
        # self.df = None
        self.name = None
        self.size = None
        self.file_path = None
        # self.counters = dict()
        self.key_hist = {}
        self.non_key_hist = {}
        self.jk_corrector = {}  # {topKey:ValueWhereChanged}
        self.categorical_hist = {}

    def fit(
        self, file, join_keys, relevant_keys, bin_info, schema: SchemaGraph, args=None
    ) -> None:
        sep = ","
        if ".dat" in file:
            sep = "|"

        if args.dataset == 'job-light':
            table_name = file.split("/")[-1].split(".")[0]
            print('read imdb')
            # list_name = schema.table_dictionary[table_name].attributes
            # df = pd.read_csv(file, names=list_name, header=None, escapechar='\\', encoding='utf-8',
            #                  quotechar='"',
            #                  sep=",")
            df = pd.read_csv(file, sep=",")
        else:
            df = pd.read_csv(file, sep=sep)
        # df = pd.read_csv(file, sep=sep)
        self.size = df.shape[0]
        self.file_path = file
        self.name = file.split("/")[-1].split(".")[0]

        # if args.cdf:
        #     self.use_cdf = args.cdf

        # currently only support at most 1 join keys in a table
        # print("join_keys", join_keys)
        # print("df", df)
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

        # print("relevant_keys", relevant_keys[self.name])
        for relev_key in relevant_keys[self.name]:
            # logger.info("col is %s", relev_key)
            # exit()
            df_col = df[[relev_key]]  # .fillna(-1)  # replace NULL with -1 !
            column = NonKeyColumnTopK()
            column.fit(df_col, args=args)
            self.non_key_hist[relev_key] = column
            # if args.cdf:
            #     self.cdfs[relev_key] = column
            # else:
            #     self.pdfs[relev_key] = column

        # if self.name in schema.categoricals:
        #     for jk in schema.categoricals[self.name]:
        #         if jk not in self.categorical_hist:
        #             self.categorical_hist[jk] = {}
        #         for cate_key in schema.categoricals[self.name][jk]:
        #             logger.info("!!!%s,%s,%s", self.name, jk, cate_key)

        #             if cate_key not in self.categorical_hist[jk]:
        #                 self.categorical_hist[jk][cate_key] = {}

        #             cate_values = df[cate_key].unique()
        #             cate_values = cate_values[~np.isnan(cate_values)]
        #             for cate_val in cate_values:
        #                 df_jk = df.loc[df[cate_key] == cate_val][[jk]]
        #                 logger.info("df_jk\n %s", df_jk)

        #                 column = KeyColumnTopK()

        #                 if self.name in bin_info and jk in bin_info[self.name]:
        #                     bins = np.linspace(
        #                         bin_info[self.name][jk][0],
        #                         bin_info[self.name][jk][1],
        #                         args.grid,
        #                     )
        #                     column.fit(df_col, bins=bins, args=args)
        #                     self.categorical_hist[jk][cate_key][cate_val] = column
        #                 else:
        #                     logger.error("bin info is not provided.")
        #                     exit()

    def fit_join_key_corrector(
        self,
        file,
        join_keys,
        relevant_keys,
        bin_info,
        top_container,
        join_path,
        jks,
        args=None,
    ):
        sep = ","
        if ".dat" in file:
            sep = "|"
        df = pd.read_csv(file, sep=sep)

        # logger.info("join keys %s", join_keys)
        # logger.info("jks %s", jks)

        # for jk  in join_keys[self.name]:
        # logger.info("relev %s", relevant_keys[self.name])
        cols = list(set(relevant_keys[self.name]) - set([jks]))
        # logger.info("cols %s", cols)
        df1 = df[[jks] + cols]
        # exit()
        # logger.info("df1 \n %s", df1)

        top_ids = list(top_container.keys())
        # logger.info("top id is %s", top_ids)
        top_ids = [int(i) for i in top_ids]
        # logger.info("top id is %s", top_ids)
        df_filtered = df1[df1[jks].isin(top_ids)]
        # logger.info("df_filtered is \n%s", df_filtered)
        cols = df_filtered.columns.to_list()
        col_key = cols[0]

        self.jk_corrector[jks] = {}
        for col in cols[1:]:
            # print(df_filtered[[col_key, col]])
            # if jks == "UserId" and col == "PostHistoryTypeId":
            #     print("df_filtered:\n", df_filtered)
            #     exit()
            d = (
                pd.Series(df_filtered[col].values, index=df_filtered[col_key])
                .dropna()
                .to_dict()
            )  # TODO check if needs to find the min or max, currently, it is random one
            self.jk_corrector[jks][col] = d
        # logger.info("jk_corrector \n%s", self.jk_corrector)

    def filter_join_key_by_query(self, domain: Domain, col: str, jks, ids: list = []):
        # logger.info("jk_corrector %s", self.jk_corrector.keys())
        # # id_considered = set(self.jk_corrector[col].keys())
        # # if ids:
        # #     id_considered = id_considered.intersection(set(ids))
        # logger.info("self.jk_corrector %s", self.jk_corrector[jks][col])
        # # exit()
        # hh = KeyColumnTopK()
        # # hh.pdf.top_k_container
        # logger.info("self.key_hist[jks] %s",
        #             self.key_hist.keys())
        # # TODO here:lambda filter
        # filtered_top_k_not_included = [
        #     list(filter(lambda x: not domain.contain(x), list(d.keys()))) for d in self.key_hist[jks].pdf.top_k_container]
        # logger.info("filtered_top_k_not_included %s",
        #             filtered_top_k_not_included)

        # exit()
        filtered_dict = {
            k: v
            for k, v in self.jk_corrector[jks][col].items()
            if not domain.contain(v)
        }

        filtered_out_id = list(filtered_dict.keys())

        filtered_out_id += ids
        uniques = list(set(filtered_out_id))
        return uniques


class KeyColumnTopK:
    pdf: UpperBoundHistogramTopK

    def __init__(self) -> None:
        self.name = None
        self.pdf = None
        self.min = None
        self.max = None

    def fit(self, df_column, bins, args=None) -> None:
        method = args.method
        if method == "topk":
            kde = UpperBoundHistogramTopK(args.topk)
        elif method == "joinhist":
            kde = JoinHistogram()
        elif method == "upperbound":
            kde = UpperBoundHistogram()
        else:
            logger.error("unexpected method type %s", method)
            exit()

        kde.fit(df_column, df_column.columns, bins=bins)
        self.pdf = kde


class NonKeyColumnTopK:
    pdf: NonKeyTopKHistogram

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
