import time

import numpy as np
import scipy.integrate as integrate

from joins.base_logger import logger
from joins.join_graph import get_join_hyper_graph
from joins.parser import parse_query_simple
from joins.schema_base import identify_conditions, identify_key_values
from joins.stats.schema import get_stats_relevant_attributes
from joins.table import TableContainer


class ApproximateEngine:
    def __init__(self,  models: dict[str, TableContainer] = None, auto_grid=True) -> None:
        self.models: dict = models
        self.auto_grid = auto_grid
        self.all_keys, self.equivalent_keys, self.table_keys = identify_key_values(
            models['schema'])
        self.join_keys, self.relevant_keys, self.counters = get_stats_relevant_attributes(
            models['schema'])

    def query(self, query_str):

        tables_all, table_queries, join_cond, join_keys = parse_query_simple(
            query_str)
        equivalent_group = get_join_hyper_graph(
            join_keys, self.equivalent_keys)
        key_conditions, non_key_conditions = identify_conditions(
            table_queries, join_keys)

        # currently, no selection on the join key
        assert (len(key_conditions) == 0)

        print("keys: \n", self.models.keys())

        print(query_str)
        # print(tables_all)
        # print("table_queries\n", table_queries)
        print("join_cond", join_cond)
        print("join_keys", join_keys)
        # print(equivalent_group)
        # exit()
        print("key_conditions\n", key_conditions)
        print("non_key_conditions\n", non_key_conditions)
        if key_conditions:
            logger.warning("selection on join key is not supported yet.")
            return
        if not non_key_conditions:  # simple case, no selections
            return simple_card(self.models, tables_all, join_cond,
                               self.relevant_keys, self.counters)
        else:
            return selection_card(self.models, tables_all, join_cond, non_key_conditions,
                                  self.relevant_keys, self.counters)
        return

    def integrate1d(self, model_name, l, h):
        if self.auto_grid:
            return integrate.quad(lambda x: self.models[model_name].predict(x), l, h)
        else:
            logger.warning("grid integral is not implemented yet")

    def integrate2d(self,):
        pass


def simple_card(models: dict[str, TableContainer], tables_all, join_cond, relevant_keys, counters, grid=10000):
    t11 = time.time()
    assert (len(join_cond) == 1)
    col_models = []
    n_models = []
    conditions = [cond.split(" = ") for cond in join_cond][0]
    print("conditions", conditions)
    print("join_cond:", join_cond)
    print("tables_all: ", tables_all)
    t1, k1 = conditions[0].split(".")
    t2, k2 = conditions[1].split(".")
    print(t1, k1)
    n1, n2 = counters[t1], counters[t2]

    mdl1, mdl2 = models[t1].pdfs[k1], models[t2].pdfs[k2]
    print(mdl1)
    mins = max(mdl1.min, mdl2.min)
    maxs = min(mdl1.max, mdl2.max)
    assert (mins < maxs)

    t2 = time.time()
    # *
    result = integrate.quad(lambda x: mdl1.pdf.predict(
        x)*mdl2.pdf.predict(x), mins, maxs, limit=500)[0] * n1*n2
    print("result: ", result)
    t3 = time.time()

    x = np.linspace(mins, maxs, grid)
    width = x[1] - x[0]
    # print("width:", width)
    pred1 = mdl1.pdf.predict(x)
    pred2 = mdl2.pdf.predict(x)
    result = width*np.sum(np.multiply(pred1, pred2))*n1*n2  # *width*n1*n2
    # result = integrate.quad(lambda x: mdl1.pdf.predict(
    #     x)*mdl2.pdf.predict(x), mins, maxs)[0]*n1*n2
    print("result: ", result)
    logger.info("time cost for this query is %f", (t3-t2))
    logger.info("time cost for this query is %f", (time.time()-t3))
    logger.info("time cost for this query is %f", (time.time()-t11))
    return result


def selection_card(models: dict[str, TableContainer], tables_all, join_cond, non_key_conditions, relevant_keys, counters, grid=10000):
    t11 = time.time()
    assert (len(join_cond) == 1)
    col_models = []
    n_models = []
    conditions = [cond.split(" = ") for cond in join_cond][0]
    logger.info("conditions %s", conditions)
    logger.info("join_cond: %s", join_cond)
    logger.info("tables_all: %s", tables_all)
    logger.info("non_key_conditions: %s", non_key_conditions)
    t1, k1 = conditions[0].split(".")
    t2, k2 = conditions[1].split(".")
    print(t1, k1)
    n1, n2 = counters[t1], counters[t2]

    mdl1, mdl2 = models[t1].pdfs[k1], models[t2].pdfs[k2]
    print(mdl1)
    mins = max(mdl1.min, mdl2.min)
    maxs = min(mdl1.max, mdl2.max)
    assert (mins < maxs)

    t2 = time.time()
    # *
    result = integrate.quad(lambda x: mdl1.pdf.predict(
        x)*mdl2.pdf.predict(x), mins, maxs, limit=500)[0] * n1*n2
    print("result: ", result)
    t3 = time.time()

    x = np.linspace(mins, maxs, grid)
    width = x[1] - x[0]
    # print("width:", width)
    pred1 = mdl1.pdf.predict(x)
    pred2 = mdl2.pdf.predict(x)
    result = width*np.sum(np.multiply(pred1, pred2))*n1*n2  # *width*n1*n2
    # result = integrate.quad(lambda x: mdl1.pdf.predict(
    #     x)*mdl2.pdf.predict(x), mins, maxs)[0]*n1*n2
    print("result: ", result)
    logger.info("time cost for this query is %f", (t3-t2))
    logger.info("time cost for this query is %f", (time.time()-t3))
    logger.info("time cost for this query is %f", (time.time()-t11))
    return result
