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

        # print("keys: \n", self.models.keys())

        print(query_str)
        # print(tables_all)
        # print("table_queries\n", table_queries)
        # print("join_cond", join_cond)
        # print("join_keys", join_keys)
        # print(equivalent_group)
        # exit()
        # print("key_conditions\n", key_conditions)
        # print("non_key_conditions\n", non_key_conditions)
        if key_conditions:
            logger.warning("selection on join key is not supported yet.")
            return
        if not non_key_conditions:  # simple case, no selections
            return simple_card(self.models, tables_all, join_cond,
                               self.relevant_keys, self.counters)
        else:
            return right_table_selection_card(self.models, tables_all, join_cond, non_key_conditions,
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
    # print("conditions", conditions)
    # print("join_cond:", join_cond)
    # print("tables_all: ", tables_all)
    t1, k1 = conditions[0].split(".")
    t2, k2 = conditions[1].split(".")
    # print(t1, k1)
    n1, n2 = counters[t1], counters[t2]

    mdl1, mdl2 = models[t1].pdfs[k1], models[t2].pdfs[k2]
    # print(mdl1)
    mins = max(mdl1.min, mdl2.min)
    maxs = min(mdl1.max, mdl2.max)
    assert (mins < maxs)

    # t2 = time.time()
    # # *
    # result = integrate.quad(lambda x: mdl1.pdf.predict(
    #     x)*mdl2.pdf.predict(x), mins, maxs, limit=500)[0] * n1*n2
    # print("result: ", result)
    # t3 = time.time()

    x = np.linspace(mins, maxs, grid)
    width = x[1] - x[0]
    # print("width:", width)
    pred1 = mdl1.pdf.predict(x)
    pred2 = mdl2.pdf.predict(x)
    result = width*np.sum(np.multiply(pred1, pred2))*n1*n2  # *width*n1*n2
    # result = integrate.quad(lambda x: mdl1.pdf.predict(
    #     x)*mdl2.pdf.predict(x), mins, maxs)[0]*n1*n2
    # print("result: ", result)
    # logger.info("time cost for this query is %f", (t3-t2))
    # logger.info("time cost for this query is %f", (time.time()-t3))
    logger.info("time cost for this query is %f", (time.time()-t11))
    return result


def right_table_selection_card(models: dict[str, TableContainer], tables_all, join_cond, non_key_conditions, relevant_keys, counters, grid=100):
    t11 = time.time()
    assert (len(join_cond) == 1)
    join_cond = [i.replace(' ', '') for i in join_cond]
    col_models = []
    n_models = []
    conditions = [cond.replace(' ', '').split(" = ") for cond in join_cond][0]
    # logger.info("conditions %s", conditions)
    # logger.info("join_cond: %s", join_cond)
    # logger.info("tables_all: %s", tables_all)
    # logger.info("non_key_conditions: %s", non_key_conditions)
    # logger.info(join_cond[0].split("."))
    join_table_and_keys = [c.split("=")
                           for c in join_cond]  # [i.split(".") for i in
    tbl_and_join_key_dict = {}
    for i in join_table_and_keys[0]:
        t_name, k_name = i.split(".")
        tbl_and_join_key_dict[t_name] = k_name
    # join_table_and_keys = [i.split(".") for i in join_table_and_keys[0]]

    # logger.info("tbl_and_join_key_dict %s", tbl_and_join_key_dict)
    # t1, k1 = join_cond[0].split(".")
    # t2, k2 = join_cond[1].split(".")
    # print(t1, k1)
    # n1, n2 = counters[t1], counters[t2]

    # only right table is allowd to have a selection condition. not left
    assert (len(non_key_conditions) == 1)

    right_table_name = list(non_key_conditions.keys())[0]
    # logger.info("right_table_name %s", right_table_name)
    # exit()
    # logger.info(models[right_table_name].correlations.keys())
    # mdl_right = models[right_table_name].correlations[tbl_and_join_key_dict[right_table_name]]
    left_table_name = list(tbl_and_join_key_dict.keys())
    left_table_name.remove(right_table_name)
    left_table_name = left_table_name[0]
    # logger.info("left_table_name %s", left_table_name)

    # get the non_key
    non_keys = non_key_conditions[right_table_name]
    # logger.info("non_keys,%s", non_keys)

    bounds = {}
    for non_k in non_keys:
        k = non_k.split(".")[1]
        domain = [-np.Infinity, np.Infinity]
        for op in non_keys[non_k]:
            if '>' in op:
                domain[0] = non_keys[non_k][op]
            elif '<' in op:
                domain[1] = non_keys[non_k][op]
            elif '=' in op:
                domain[0], domain[1] = non_keys[non_k][op] - \
                    0.5, non_keys[non_k][op]+0.5
        bounds[k] = domain

    assert (len(bounds) == 1)  # only one selection condition at this moment.
    extra_col = list(bounds.keys())[0]

    mdl1, mdl2 = models[left_table_name].pdfs[tbl_and_join_key_dict[left_table_name]
                                              ], models[right_table_name].correlations[tbl_and_join_key_dict[right_table_name]][extra_col]
    # print(mdl1)
    # print("mdl2", mdl2)
    # print("mdl2.min", mdl2.min)
    mins = max(mdl1.min, mdl2.min[0])
    maxs = min(mdl1.max, mdl2.max[0])
    assert (mins < maxs)
    n1, n2 = counters[left_table_name], counters[right_table_name]

    t2 = time.time()
    # *
    bound = bounds[extra_col]
    if mdl2.min[1] > bound[0]:
        bound[0] = mdl2.min[1]
    if mdl2.max[1] < bound[1]:
        bound[1] = mdl2.max[1]
    # logger.info("bound is %s", bound)

    # def f(y, x): return mdl1.pdf.predict(x)*mdl2.pdf.predict([x, y])
    # result = integrate.dblquad(
    #     f, mins, maxs, bound[0], bound[1])[0] * n1*n2
    # print("result: ", result)
    t3 = time.time()

    grid_x_size = 300
    grid_y_size = 300
    x_grid = np.linspace(mins, maxs, grid_x_size)
    width_x = x_grid[1] - x_grid[0]
    y_grid = np.linspace(bound[0], bound[1], grid_y_size)
    width_y = y_grid[1] - y_grid[0]
    # print("width:", width)
    pred1 = mdl1.pdf.predict(x_grid)

    # result1 = np.sum(pred1)*width_x
    pred2 = mdl2.pdf.predict_grid(x_grid, y_grid)

    # print("pred1", pred1)
    pred1 = np.array([pred1 for _ in range(grid_y_size)]
                     ).reshape(grid_x_size, grid_y_size)
    # pred1 = np.repeat([pred1], 20).reshape(50, 20)

    # print("pred1", pred1)
    # print("pred1.repeat", np.repeat(pred1, grid_y_size))
    # print("pred2", pred2)
    # print("pred1", pred1.shape)
    # print("pred2", pred2.shape)
    # result2 = width_x*width_y * \
    #     np.sum(pred2)  # *width*n1*n2
    result = np.sum(np.multiply(pred1, pred2))*width_x*width_y*n1*n2
    logger.info("result: %f", result)
    # print("result2", result2)
    # result = integrate.quad(lambda x: mdl1.pdf.predict(
    #     x)*mdl2.pdf.predict(x), mins, maxs)[0]*n1*n2
    # print("result: ", result)
    # logger.info("time cost for this query is %f", (t3-t2))
    # logger.info("time cost for this query is %f", (time.time()-t3))
    logger.info("time cost for this query is %f", (time.time()-t11))
    return result
