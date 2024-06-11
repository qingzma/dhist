# this is an example for p(yz|x)= p(y|x)p(z|x), assuming y and z are conditionally independent given x
# https://stats.stackexchange.com/questions/29510/proper-way-to-combine-conditional-probability-distributions-of-the-same-random-v
import copy
import math
import time
from enum import Enum

import numpy as np
import scipy.integrate as integrate

from joins.base_logger import logger
from joins.domain import (
    Domain,
    JoinKeysGrid,
    SingleTablePushedDownCondition,
    generate_push_down_conditions,
    get_idx_in_lists,
)
from joins.histograms.histograms import (
    UpperBoundHistogramTopK,
    get_dominating_items_in_histograms,
)
from joins.join_graph import get_join_hyper_graph
from joins.parser import get_max_dim, get_two_chained_query, parse_query_simple
from joins.plots import plot_line, plot_vec_sel_array
from joins.schema_base import SchemaGraph, identify_conditions, identify_key_values
from joins.stats.schema import get_stats_relevant_attributes
from joins.table import Column, Column2d, TableContainer
from joins.table_top_k import TableContainerTopK


class EngineTopK:
    def __init__(
        self, models: dict[str, TableContainerTopK] = None, auto_grid=True, use_cdf=True
    ) -> None:
        self.models: dict = models
        self.auto_grid = auto_grid
        self.all_keys, self.equivalent_keys, self.table_keys = identify_key_values(
            models["schema"]
        )
        (
            self.join_keys,
            self.relevant_keys,
            self.counters,
        ) = get_stats_relevant_attributes(models["schema"])
        self.grid_size_x = 1000
        self.grid_size_y = 1000
        self.use_cdf = use_cdf

    def query(self, query_str, n_dominating_counter=-1):
        logger.info("QUERY [%s]", query_str)
        tables_all, table_query, join_cond, join_keys = parse_query_simple(query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys
        )
        # max_dim = get_max_dim(join_keys)

        # logger.info("join_cond is %s", join_cond)
        # logger.info("tables_all is %s", tables_all)
        # logger.info("table_query is %s", table_query)
        # # logger.info("join_keys is %s", join_keys)
        # logger.info("conditions %s", conditions)
        # logger.info("max_dim %s", max_dim)
        # logger.info("join_paths %s", join_paths)

        # single table query
        if len(tables_all) == 1:
            res = single_table_query(self.models, conditions)
            return res

        # join query
        join_paths = parse_join_paths(join_cond)
        max_dim = len(join_paths)

        # only a single join key is allowed in a table
        if max_dim == 2:
            return 2

        if max_dim > 2:
            logger.error("length is  [%i]", max_dim)
            logger.error("is 3 [%s]", query_str)
            exit()

        # for cases with max_dim==1

        res = multi_query_with_same_column(
            self.models, conditions, join_cond, join_paths, n_dominating_counter
        )

        return res


def single_table_query(models: dict[str, TableContainerTopK], conditions):
    # logger.info("models are %s", models)
    # logger.info("condtions are %s", conditions)
    assert len(conditions) == 1
    tbl = list(conditions.keys())[0]
    conds = conditions[tbl]

    if len(conds) == 1 and conds[0].non_key is None:
        return models[tbl].size

    selectivity = 1.0
    for cond in conds:
        model = models[tbl].non_key_hist[cond.non_key.split(".")[1]].pdf
        # logger.info("cond is %s", cond.non_key_condition)
        domain_query = cond.non_key_condition
        selectivity *= model.selectivity(domain_query)
        # logger.info("cond is %s", domain_query)
        # logger.info("selectivity is %s", model.selectivity(domain_query))
    return models[tbl].size * selectivity


def parse_join_paths(join_cond: list):
    paths = []
    for cond in join_cond:
        cond_no_space = cond.replace(" ", "")
        splits = cond_no_space.split("=")

        b_found = False
        for path in paths:
            if splits[0] in path:
                path.add(splits[1])
                b_found = True
                break
            if splits[1] in path:
                path.add(splits[0])
                b_found = True
                break
        if not b_found:
            paths.append(set([splits[0], splits[1]]))
    pathss = [list(p) for p in paths]
    # logger.info("paths are %s", pathss)

    # path_sorted = []
    for ps in pathss:
        ps.sort(key=lambda x: (".Id" in x, x))

    # logger.info("paths are %s", pathss)
    return pathss


def multi_query_with_same_column(
    models: dict[str, TableContainerTopK],
    conditions,
    join_cond,
    join_paths,
    n_dominating_counter=-1,
):
    conditions = copy.deepcopy(conditions)
    # filter top k dominating items, which will be exclude by range predicates.
    id_filtered_out = []
    for tbl in conditions:  # ["users"]:  # conditions:  # conditions:
        for cond in conditions[tbl]:
            if cond.non_key is not None:
                # logger.info("model predicates %s", models[tbl].jk_corrector)
                domain_query = cond.non_key_condition
                # logger.info("join key is %s", cond.join_keys)
                jks = cond.join_keys[0].split(".")[1]
                # logger.info("jks is %s", jks)
                # exit()
                if (
                    jks == "Id"
                ):  # TODO this is a tempary soluiton to identify the primary table
                    id_filtered_out = models[tbl].filter_join_key_by_query(
                        domain_query,
                        col=cond.non_key.split(".")[1],
                        jks=jks,
                        ids=id_filtered_out,
                    )
    # logger.info("final id is %s", id_filtered_out)

    splits = join_paths[0][0].split(".")
    tbl = splits[0]
    jk = splits[1]
    hist: UpperBoundHistogramTopK = models[tbl].key_hist[jk].pdf
    # logger.info("join_paths %s", join_paths)
    # sg = SchemaGraph()
    # sg.categoricals
    # con = SingleTablePushedDownCondition()
    # cond
    schema = models["schema"]
    # logger.info("model schema %s", models["schema"])
    # exit()
    # check for categorical attribute
    cate_hists = {}
    for tbl in conditions:
        for cond in conditions[tbl]:
            if cond.non_key is not None:
                relev_key = cond.non_key.split(".")[1]
                join_key = cond.join_keys[0].split(".")[1]
                relev_domain = cond.non_key_condition

                # logger.info("hahaha %s, %s, %s", join_key, relev_key, relev_domain)

                if (
                    tbl in schema.categoricals
                    and join_key in schema.categoricals[tbl]
                    and relev_key in schema.categoricals[tbl][join_key]
                ):
                    # logger.info("non_key_condition %s", relev_domain)
                    if relev_domain.min == relev_domain.max:
                        relev_val = relev_domain.min
                        # logger.info("relev_val %s", relev_val)
                        col_model = models[tbl].categorical_hist[join_key][relev_key][
                            relev_val
                        ]
                        cate_hists["-".join([tbl, join_key])] = col_model
                        # conds = copy.deepcopy(conditions[tbl])
                        # conds.remove(cond)
                        # conditions[tbl] = conds
                        # TODO implement this remove funciton here
    for table_join_key in join_paths[0][1:-1]:
        splits1 = table_join_key.split(".")
        tbl1 = splits1[0]
        jk1 = splits1[1]

        cate_hist_key = "-".join([tbl1, jk1])
        # logger.info("cate_hist_key %s", cate_hist_key)
        # logger.info("cate_hists %s", cate_hists)

        hist1 = (
            cate_hists[cate_hist_key].pdf
            if cate_hist_key in cate_hists
            else models[tbl1].key_hist[jk1].pdf
        )

        hist = hist.join(hist1)  # , update_statistics=True

    splits1 = join_paths[0][-1].split(".")
    tbl1 = splits1[0]
    jk1 = splits1[1]
    cate_hist_key = "-".join([tbl1, jk1])
    hist1 = (
        # cate_hists[cate_hist_key].pdf
        # if cate_hist_key in cate_hists
        # else
        models[tbl1]
        .key_hist[jk1]
        .pdf
    )
    res = hist.join(hist1, id_filtered=id_filtered_out)

    selectivity = 1.0
    for tbl in conditions:
        for cond in conditions[tbl]:
            if cond.non_key is not None:
                model = models[tbl].non_key_hist[cond.non_key.split(".")[1]].pdf
                domain_query = cond.non_key_condition
                selectivity *= model.selectivity(domain_query)
                # logger.info("selectivity is %s", model.selectivity(domain_query))

    # logger.info("top k is %s", res.top_k_container)
    if n_dominating_counter > 0:
        return (
            get_dominating_items_in_histograms(
                res.top_k_container, n=n_dominating_counter, size=np.sum(res.counts)
            ),
            join_paths[0],
        )
    return np.sum(res.counts) * selectivity
