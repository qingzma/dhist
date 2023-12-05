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
                self.models, conditions, self.grid_size_x, self.grid_size_y)
            tbl = list(conditions.keys())[0]
            n = self.models[tbl].size
            return np.sum(vec_sel)*n


def vec_sel_single_table_query(models: dict[str, TableContainer], conditions: list[SingleTablePushedDownCondition], grid_size_x, grid_size_y, use_column_model=True):
    assert (len(conditions) == 1)
    tbl = list(conditions.keys())[0]
    conds = conditions[tbl]
    # logger.info("conds: %s", conds)
    if len(conds) == 1:
        cond = conds[0]
        # no selection, simple cardinality, return n
        # [SingleTablePushedDownCondition[badges]--join_keys[badges.Id]--non_key[None]--condition[None, None]--to_join[{}]]]
        if cond.non_key is None:
            return np.array([1.0])  # [models[tbl].size]

        # one selection
