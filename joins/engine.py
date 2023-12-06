# this is an example for p(yz|x)= p(y|x)p(z|x), assuming y and z are conditionally independent given x
# https://stats.stackexchange.com/questions/29510/proper-way-to-combine-conditional-probability-distributions-of-the-same-random-v
import time
from enum import Enum

import numpy as np
import scipy.integrate as integrate

from joins.base_logger import logger
from joins.domain import (Domain, JoinKeysGrid, SingleTablePushedDownCondition,
                          generate_push_down_conditions, get_idx_in_lists)
from joins.join_graph import get_join_hyper_graph
from joins.parser import parse_query_simple
from joins.schema_base import identify_conditions, identify_key_values
from joins.stats.schema import get_stats_relevant_attributes
from joins.table import Column, Column2d, TableContainer


class Engine:
    def __init__(self,  models: dict[str, TableContainer] = None, auto_grid=True, use_cdf=True) -> None:
        self.models: dict = models
        self.auto_grid = auto_grid
        self.all_keys, self.equivalent_keys, self.table_keys = identify_key_values(
            models['schema'])
        self.join_keys, self.relevant_keys, self.counters = get_stats_relevant_attributes(
            models['schema'])
        self.grid_size_x = 1000
        self.grid_size_y = 1000
        self.use_cdf = use_cdf

    def query(self, query_str):
        logger.info("QUERY [%s]", query_str)
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys)

        # single table query
        if len(tables_all) == 1:
            vec_sel = vec_sel_single_table_query(
                self.models, conditions, self.grid_size_x)
            tbl = list(conditions.keys())[0]
            n = self.models[tbl].size
            return np.sum(vec_sel)*n


def vec_sel_single_table_query(models: dict[str, TableContainer], conditions: list[SingleTablePushedDownCondition], grid_size_x, use_column_model=False):
    assert (len(conditions) == 1)
    tbl = list(conditions.keys())[0]
    conds = conditions[tbl]

    # sz_min = np.Infinity
    # logger.info("conds: %s", conds)
    if len(conds) == 1:
        cond = conds[0]
        # no selection, simple cardinality, return n
        # [SingleTablePushedDownCondition[badges]--join_keys[badges.Id]--non_key[None]--condition[None, None]--to_join[{}]]]
        if cond.non_key is None:
            sz_min = models[tbl].size
            return np.array([1.0])  # [models[tbl].size]

        # one selection
        if use_column_model:
            logger.info("models[tbl].pdfs %s", models[tbl].pdfs.keys())
            model: Column = models[tbl].cdfs[cond.non_key.split(
                ".")[1]] if models[tbl].use_cdf else models[tbl].pdfs[cond.non_key.split(".")[1]]
            # logger.info("model is %s", model)
            domain = Domain(model.min, model.max, True, True)
            domain_query = cond.non_key_condition
            domain.merge_domain(domain_query)
            pred = model.pdf.predict_domain(domain)
            logger.info("selectivity is %s", np.sum(pred))
            return pred
        # for cases if column model is not used.
        model: Column2d = models[tbl].correlations_cdf["Id"][cond.non_key.split(".")[
            1]]
        jk_domain = [model.min[0], model.max[0]]
        nk_domain = Domain(model.min[1], model.max[1], True, True)
        nk_domain_query = cond.non_key_condition
        # logger.info("nk_domain_query %s", nk_domain_query)
        nk_domain.merge_domain(nk_domain_query)
        # logger.info("nk_domain %s", nk_domain)

        grid_x, width_x = np.linspace(
            *jk_domain, grid_size_x, retstep=True)
        # grid_y, width_y = np.linspace(
        #     nk_domain.min, nk_domain.max, grid_size_y, retstep=True)
        pred = model.pdf.predict_grid_with_y_range(grid_x, nk_domain)
        # logger.info("pred is %s", pred)
        # logger.info("sum is %s", np.sum(pred))
        # logger.info("sums is %s", np.sum(pred)*width_x)
        return pred*width_x  # , model.size
    # multiple selection
    cond0 = conds[0]
    jk = cond0.join_keys[0].split(".")[1]
    jk_model = models[tbl].pdfs[jk]
    jk_domain = [jk_model.min, jk_model.max]
    # logger.info("x range is %s", jk_domain)
    grid_x, width_x = np.linspace(
        *jk_domain, grid_size_x, retstep=True)
    pred_x = vec_sel_single_column(jk_model, grid_x)

    pred_xys = []

    for condi in conds:
        sel = vec_sel_single_table_query(
            models, {tbl: [condi]}, grid_size_x, use_column_model=False)
        pred_xys.append(sel)
        logger.info("sub selectivity is %s", np.sum(sel))

        # logger.info("size is %s", sz)
        # if sz < sz_min:
        #     sz_min = sz

    # logger.info("predx is %s", np.sum(pred_x))
    pred = np.ones_like(pred_x)
    for pred_xyi in pred_xys:
        pred = vec_sel_multiply(pred, vec_sel_divide(pred_xyi, pred_x))/width_x

    # logger.info("width x is %s", width_x)
    res = width_x*vec_sel_multiply(pred, pred_x)
    return res  # , sz_min


def vec_sel_single_column(column, x_grid):
    return column.pdf.predict(x_grid)


def vec_sel_multiply(sel1, sel2):
    return np.multiply(sel1, sel2)


def vec_sel_divide(sel1, sel2):
    # for 0 division, force to zero
    return np.divide(sel1, sel2,  out=np.zeros_like(
        sel1), where=sel2 != 0)
