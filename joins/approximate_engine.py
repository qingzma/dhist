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


class QueryType(Enum):
    TwoTableNoSelection = 1
    TwoTableRightSelection = 2
    MultiTableSingleJoinKeyNoSelection = 3
    MultiTableMultiJoinKeyNoSelection = 4
    SelectionOnJoinKey = 5
    SINGLE_TABLE_SINGLE_JK_NO_SELECTION = 6
    SINGLE_TABLE_SINGLE_JK_SINGLE_SELECTION = 7
    SINGLE_TABLE_SINGLE_JK_MULTI_SELECTION = 8
    SINGLE_TABLE_MULTI_JOIN_KEY = 9
    NotSupported = 20


class ApproximateEngine:
    def __init__(self,  models: dict[str, TableContainer] = None, auto_grid=True, use_cdf=False) -> None:
        self.models: dict = models
        self.auto_grid = auto_grid
        self.all_keys, self.equivalent_keys, self.table_keys = identify_key_values(
            models['schema'])
        self.join_keys, self.relevant_keys, self.counters = get_stats_relevant_attributes(
            models['schema'])
        self.grid_size_x = 200
        self.grid_size_y = 200
        self.use_cdf=use_cdf

    def query_with_pushed_down(self, query_str):
        logger.info("QUERY [%s]", query_str)
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
        # logger.info("tables_all %s", tables_all)
        # logger.info("table_query %s", table_query)
        # logger.info("join_cond %s", join_cond)
        # logger.info("join_keys %s", join_keys)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys)

        # single table query
        if len(tables_all) == 1:
            return process_single_table_query(self.models, conditions, self.grid_size_x, self.grid_size_y, use_cdf=self.use_cdf)

        # join query
        join_keys_grid = JoinKeysGrid()
        join_keys_grid.calculate_push_down_join_keys_domain(
            conditions, join_cond, self.models, tables_all, self.grid_size_x)

        # only a single join key is allowed in a table
        assert (len(join_keys_grid.join_keys_domain) == 1)
        # join_keys_lists, join_keys_domain = calculate_push_down_join_keys_domain(
        #     conditions, join_cond, self.models, tables_all)

        n = get_cartesian_cardinality(self.counters, tables_all)
        pred = process_push_down_conditions(
            self.models, conditions, join_cond, join_keys_grid,use_cdf=self.use_cdf)

        logger.info("cartesian is %E", n)
        logger.info("pred is %s ", pred)

        return pred*n

    def query(self, query_str):
        logger.info("QUERY [%s]", query_str)
        query_type, tables_all, join_cond, non_key_conditions, tbl_and_join_key_dict, join_key_in_each_table = parse_query_type(
            query_str, self.equivalent_keys)
        if query_type == QueryType.SINGLE_TABLE_SINGLE_JK_NO_SELECTION:
            return single_table_count(self.models[next(iter(tables_all.values()))])

        if query_type == QueryType.SelectionOnJoinKey:
            logger.warning(
                "selection on join key is not supported yet, but coule be easily supported.")
            return
        if query_type == QueryType.TwoTableNoSelection:
            return simple_card(self.models, tables_all, join_cond,
                               self.relevant_keys, self.counters)
        if query_type == QueryType.TwoTableRightSelection:
            return right_table_selection_card(self.models, tables_all, join_cond, non_key_conditions,
                                              self.relevant_keys, self.counters)
        if query_type == QueryType.MultiTableSingleJoinKeyNoSelection:
            return multiple_table_same_join_column(self.models, tables_all, join_cond, non_key_conditions, tbl_and_join_key_dict, join_key_in_each_table, self.relevant_keys, self.counters)
        else:
            logger.info("QueryType: %s", query_type)
            logger.warning("not implemented yet")
            return


def single_table_count(table: TableContainer):
    return table.size


def simple_card(models: dict[str, TableContainer], tables_all, join_cond, relevant_keys, counters, grid=1000):
    # t11 = time.time()
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
    # logger.info("result: %f", result)
    # logger.info("time cost for this query is %f", (t3-t2))
    # logger.info("time cost for this query is %f", (time.time()-t3))
    # logger.info("time cost for this query is %f", (time.time()-t11))
    return result


def right_table_selection_card(models: dict[str, TableContainer], tables_all, join_cond, non_key_conditions, relevant_keys, counters, grid=100):
    # t11 = time.time()
    assert (len(join_cond) == 1)

    tbl_and_join_key_dict, join_key_in_each_table = get_tbl_and_join_key_dict(
        join_cond)
    # logger.info("tbl_and_join_key_dict %s", tbl_and_join_key_dict)

    assert (len(non_key_conditions) == 1)

    right_table_name = list(non_key_conditions.keys())[0]
    left_table_name = list(tbl_and_join_key_dict.keys())
    left_table_name.remove(right_table_name)
    left_table_name = left_table_name[0]

    # get the non_key
    non_keys = non_key_conditions[right_table_name]

    bounds = get_bounds(non_keys)

    assert (len(bounds) == 1)  # only one selection condition at this moment.
    extra_col = list(bounds.keys())[0]

    mdl1 = models[left_table_name].pdfs[tbl_and_join_key_dict[left_table_name][0]]
    mdl2 = models[right_table_name].correlations[tbl_and_join_key_dict[right_table_name][0]][extra_col]
    mins = max(mdl1.min, mdl2.min[0])
    maxs = min(mdl1.max, mdl2.max[0])
    assert (mins < maxs)
    n1, n2 = counters[left_table_name], counters[right_table_name]

    bound = bounds[extra_col]
    if mdl2.min[1] > bound[0]:
        bound[0] = mdl2.min[1]
    if mdl2.max[1] < bound[1]:
        bound[1] = mdl2.max[1]

    grid_x_size = 300
    grid_y_size = 200
    # t2 = time.time()
    x_grid = np.linspace(mins, maxs, grid_x_size)
    width_x = x_grid[1] - x_grid[0]
    y_grid = np.linspace(bound[0], bound[1], grid_y_size)
    width_y = y_grid[1] - y_grid[0]

    # t3 = time.time()
    pred1 = mdl1.pdf.predict(x_grid)
    # t4 = time.time()

    pred2 = mdl2.pdf.predict_grid(x_grid, y_grid)
    # t5 = time.time()

    pred1 = np.array([pred1 for _ in range(grid_y_size)]
                     ).reshape(grid_x_size, grid_y_size)

    result = np.sum(np.multiply(pred1, pred2))*width_x*width_y*n1*n2
    # t6 = time.time()
    # logger.info("time cost for grid %f", (t3-t2))
    # logger.info("time cost for 1d predict %f", (t4-t3))
    # logger.info("time cost for 2d predict is %f", (t5-t4))
    # logger.info("time cost for integral is %f", (t6-t5))

    # logger.info("result: %f", result)
    # logger.info("time cost for this query is %f", (time.time()-t11))
    return result


def multiple_table_same_join_column(models: dict[str, TableContainer], tables_all, join_cond, non_key_conditions, tbl_and_join_key_dict, join_key_in_each_table, relevant_keys, counters, grid=100):
    # logger.info(query_type, tables_all, join_cond, non_key_conditions)
    # logger.info("tables_all %s", tables_all)
    # logger.info("join_cond %s", join_cond)
    # logger.info("non_key_conditions %s", non_key_conditions)
    # logger.info("tbl_and_join_key_dict %s", tbl_and_join_key_dict)
    # logger.info("join_key_in_each_table %s", join_key_in_each_table)

    same_col_join_model_container = []
    cnt = 1
    for t in tbl_and_join_key_dict:
        col = tbl_and_join_key_dict[t][0]
        same_col_join_model_container.append(models[t].pdfs[col])
        cnt *= models[t].size
    # logger.info("models %s", same_col_join_model_container)
    # logger.info("cnt %s", cnt)
    result = cnt * \
        intergrate_1d_multi_table_same_join_key(same_col_join_model_container)
    return result


def parse_query_type(query_str, equivalent_keys):
    tables_all, table_queries, join_cond, join_keys = parse_query_simple(
        query_str)
    if not table_queries and len(tables_all) == 1:
        return QueryType.SINGLE_TABLE_SINGLE_JK_NO_SELECTION, tables_all, join_cond, None, None, None

    equivalent_group = get_join_hyper_graph(
        join_keys, equivalent_keys)
    key_conditions, non_key_conditions = identify_conditions(
        table_queries, join_keys)
    tbl_and_join_key_dict, join_key_in_each_table = get_tbl_and_join_key_dict(
        join_cond)

    if key_conditions:
        return QueryType.SelectionOnJoinKey, tables_all, join_cond, non_key_conditions, tbl_and_join_key_dict, join_key_in_each_table

    # currently, no selection on the join key
    # assert (len(key_conditions) == 0)

    if not non_key_conditions:  # simple case, no selections
        # a table has at most 1 join key
        if max([len(val)for val in tbl_and_join_key_dict.values()]) == 1:
            if len(tables_all) == 2:
                return QueryType.TwoTableNoSelection, tables_all, join_cond, non_key_conditions, tbl_and_join_key_dict, join_key_in_each_table
            else:
                return QueryType.MultiTableSingleJoinKeyNoSelection, tables_all, join_cond, non_key_conditions, tbl_and_join_key_dict, join_key_in_each_table
        else:  # exist table with two or more join keys
            logger.warning("not implemented yet")
            return
    else:
        return QueryType.TwoTableRightSelection, tables_all, join_cond, non_key_conditions, tbl_and_join_key_dict, join_key_in_each_table


def get_tbl_and_join_key_dict(join_cond):
    join_cond = [i.replace(' ', '') for i in join_cond]
    join_table_and_keys = [c.split("=")
                           for c in join_cond]  # [i.split(".") for i in
    tbl_and_join_key_dict = {}
    # logger.info("join_table_and_keys %s", join_table_and_keys)

    # if len(join_table_and_keys) == 1:  # simple 2 table join
    #     for i in join_table_and_keys[0]:
    #         t_name, k_name = i.split(".")
    #         tbl_and_join_key_dict[t_name] = k_name
    #     return tbl_and_join_key_dict

    for condi in join_table_and_keys:
        for kv in condi:
            t_name, k_name = kv.split(".")
            if t_name not in tbl_and_join_key_dict:
                tbl_and_join_key_dict[t_name] = set([k_name])
            else:
                tbl_and_join_key_dict[t_name].add(k_name)
    for t_name in tbl_and_join_key_dict:
        tbl_and_join_key_dict[t_name] = list(tbl_and_join_key_dict[t_name])

    # logger.info("tbl_and_join_key_dict %s", tbl_and_join_key_dict)
    # exit()

    join_key_in_each_table = {}
    for [i1, i2] in join_table_and_keys:
        # print("join_table_and_keys", join_table_and_keys)
        # for [i1, i2] in cond:
        if i1 not in join_key_in_each_table:
            join_key_in_each_table[i1] = [i2]
        else:
            join_key_in_each_table[i1].append(i2)

        if i2 not in join_key_in_each_table:
            join_key_in_each_table[i2] = [i1]
        else:
            join_key_in_each_table[i2].append(i1)
    # logger.info("join_key_in_each_table %s", join_key_in_each_table)

    # max_occurence = max([len(val)for val in join_key_in_each_table.values()])
    # max_keys = [k for k in join_key_in_each_table if len(
    #     join_key_in_each_table[k]) == max_occurence]

    # print("max_keys", max_keys)
    # print("max_occurence", max_occurence)

    return tbl_and_join_key_dict, join_key_in_each_table
    # exit()


def get_bounds(non_keys):
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
    return bounds


def intergrate_1d_multi_table_same_join_key(models: list[Column], grid_size=10000):
    mins = [m.min for m in models if isinstance(m, Column)]
    maxs = [m.max for m in models if isinstance(m, Column)]
    mins = max(mins)-0.5
    maxs = min(maxs)+0.5

    xs, width = np.linspace(mins, maxs, grid_size, retstep=True)
    predictions = []
    for m in models:
        predictions.append(m.pdf.predict(xs)*width)
    pred0 = predictions[0]
    if len(predictions) > 1:
        for p in predictions[1:]:
            pred0 = np.multiply(pred0, p)

    result = np.sum(pred0)
    return result


def selectivity_array_single_column(column, x_grid, grid_width):
    return column.pdf.predict(x_grid)  # *grid_width


def selectivity_array_two_columns(column, key_grid, non_key_grid, key_grid_width, non_key_grid_width):
    grid = column.pdf.predict_grid(
        key_grid, non_key_grid)
    # print(grid)
    # print("grid shape is ", len(grid), len(grid[0]))
    gg = grid*non_key_grid_width  # .reshape(len(key_grid), len(non_key_grid))
    pred = np.sum(gg, axis=1)  # *key_grid_width
    # print("pred shape is ", len(pred))
    return pred


def selectivity_grid_two_join_key(column, key1_grid, key2_grid, key1_grid_width, key2_grid_width):
    return column.pdf.predict_grid(
        key1_grid, key2_grid)  # *key1_grid_width*key2_grid_width


def combine_selectivity_array(sg1, sg2):
    return np.multiply(sg1, sg2)


def combine_selectivity_grid_with_two_arrays(arr1, grid, arr2):
    return np.multiply(np.multiply(arr1[np.newaxis, :], grid), arr2.T[:, np.newaxis])


def grid_multiply_array(grid, arr):
    tmp = np.multiply(grid, arr.T[:, np.newaxis])
    return np.sum(tmp, axis=0)


def array_multiply_grid(arr, grid):
    tmp = np.multiply(arr[np.newaxis, :], grid)
    return np.sum(tmp, axis=0)


def process_push_down_conditions(models, conditions, join_cond, join_keys_grid: JoinKeysGrid, grid_size_x=2000, grid_size_y=1000,use_cdf=False):
    logger.info("conditions: %s", conditions)
    ps = {}
    widths = {}
    for tbl in conditions:
        predictions_within_table = []
        # for condition in conditions[tbl]:
        #     pred_p = process_single_table_push_down_condition_for_join(
        #         tbl, models, condition, join_keys_grid, grid_size_y)
        #     assert (pred_p is not None)
        #     predictions_within_table.append(pred_p)

        # # get grid width for this single table
        # p = merge_single_table_predictions(
        #     conditions, predictions_within_table,)
        pred_p, width_p = process_single_table_push_down_condition_for_join(
            tbl, models, conditions[tbl], join_keys_grid, grid_size_y,use_cdf=use_cdf)
        ps[tbl] = pred_p
        widths[tbl] = width_p
    pred = merge_predictions(ps, conditions, join_cond,
                             join_keys_grid, grid_size_x)
    return pred


def process_push_down_condition(models: dict[str, TableContainer], condition: SingleTablePushedDownCondition, grid_size_x, grid_size_y, join_keys_grid: JoinKeysGrid):
    logger.debug("processing condition %s", condition)
    assert (len(condition.join_keys) == 1)
    jk = condition.join_keys[0].split(".")[1]

    # # SingleTablePushedDownCondition[badges]--join_keys[badges.Id]--non_key[None]--condition[None, None]--to_join[{}]]
    # if not condition.to_join and not condition.non_key_condition:
    #     return 1

    # # need another single talbe case
    # # SingleTablePushedDownCondition[posts]--join_keys[posts.Id]--non_key[posts.AnswerCount]--condition[0, 4]--to_join[{}]]
    # if not condition.to_join:
    #     n_key = condition.non_key.split(".")[1]
    #     model: Column2d = models[condition.tbl].correlations[jk][n_key]
    #     jk_domain = [model.min[0], model.max[0]]
    #     nk_domain_data = [model.min[1], model.max[1]]
    #     nk_domain_query = condition.non_key_condition
    #     nk_domain = merge_domain(nk_domain_data, nk_domain_query)
    #     grid_x, width_x = np.linspace(*jk_domain, grid_size_x, retstep=True)
    #     grid_y, width_y = np.linspace(*nk_domain, grid_size_y, retstep=True)

    #     pred = selectivity_array_two_columns(
    #         model, grid_x, grid_y, width_x, width_y)

    #     return pred

    idx = get_idx_in_lists(
        condition.join_keys[0], join_keys_grid.join_keys_lists)  # TODO, here only one join key is supported
    logger.info("key is %s", condition.join_keys[0])
    logger.info("join_keys_lists %s", join_keys_grid.join_keys_lists)
    # exit()
    assert (idx >= 0)
    jk_domain = join_keys_grid.join_keys_domain[idx]
    # SingleTablePushedDownCondition[comments]--join_keys[comments.UserId]--non_key[comments.Score]--condition[0, 10]--to_join[{'users': ['Id']}]]
    if condition.join_keys and condition.non_key:
        n_key = condition.non_key.split(".")[1]
        model: Column2d = models[condition.tbl].correlations[jk][n_key]

        # jk_domain = [model.min[0], model.max[0]]
        nk_domain = Domain(model.min[1], model.max[1])
        nk_domain_query = Domain(*condition.non_key_condition)
        nk_domain.merge_domain(nk_domain_query)
        grid_x, width_x = np.linspace(
            jk_domain.min, jk_domain.max, grid_size_x, retstep=True)
        grid_y, width_y = np.linspace(
            nk_domain.min, nk_domain.max, grid_size_y, retstep=True)

        pred = selectivity_array_two_columns(
            model, grid_x, grid_y, width_x, width_y)

        return pred

    # SingleTablePushedDownCondition[users]--join_keys[users.Id]--non_key[None]--condition[None, None]--to_join[{'comments': ['UserId'], 'badges': ['UserId']}]]
    if condition.join_keys and condition.non_key is None:
        model: Column2d = models[condition.tbl].pdfs[jk]
        # jk_domain = [model.min, model.max]
        grid_x, width_x = np.linspace(
            jk_domain.min, jk_domain.max, grid_size_x, retstep=True)
        pred = selectivity_array_single_column(model, grid_x, width_x)
        return pred
    return


def merge_single_table_predictions(conditions, predictions_within_table):
    logger.info("in merge table prediction")
    logger.info("len is %s", len(predictions_within_table))
    if len(predictions_within_table) == 1:
        return predictions_within_table

    pred = predictions_within_table[0]

    for pred_i in predictions_within_table[1:]:
        pred = np.multiply(pred_i, pred)
    return pred

    logger.warning("this merge method is not implemented yet.")

    return


def merge_predictions(ps, conditions, join_cond, join_keys_grid: JoinKeysGrid, grid_size_1d):
    if not join_cond:
        k = list(ps.keys())[0]
        return np.sum(ps[k])
    pred = None
    join_cond_copy = join_cond.copy()
    width = join_keys_grid.join_keys_grid[0].width

    for tbl in ps:
        logger.info(
            "table selectivity is [%s]: %s", tbl, np.sum(ps[tbl])*width)
        if pred is None:
            pred = ps[tbl]
        else:
            pred = combine_selectivity_array(pred, ps[tbl])
            logger.info(
                "table selectivity merged to [%s]: %s", tbl, np.sum(pred)*width)
    return np.sum(pred)*width

    # logger.info("in merge_predictions")
    # logger.info("len is %s", len(ps))
    while len(join_cond_copy) > 0:
        join_cond = join_cond_copy.pop()
        tk1, tk2 = join_cond.replace(" ", "").split("=")
        t1, k1 = tk1.split(".")
        t2, k2 = tk2.split(".")
        # logger.info("--ps[t1], %s", ps[t1])
        # logger.info('--sum is %s', np.sum(ps[t1]))  # TODO sum greater than 1
        if pred is None:
            pred = combine_selectivity_array(ps[t1], ps[t2])
            logger.info('--sum1 is %s', np.sum(ps[t1])*width)
            logger.info('--sum2 is %s', np.sum(ps[t2])*width)
            logger.info('--sum3 is %s', np.sum(pred)*width)
        else:
            pred = combine_selectivity_array(ps[t1], pred)
            logger.info('--sums4 is %s', np.sum(pred)*width)
    logger.info("join_keys_lists %s", join_keys_grid.join_keys_lists)
    pred = np.sum(pred)*width
    return pred


def process_single_table_query(models: dict[str, TableContainer], conditions: list[SingleTablePushedDownCondition], grid_size_x, grid_size_y, use_column_model=True, use_cdf=False):
    # logger.info("conditions: %s", conditions)
    # logger.info("len: %s", len(conditions))
    assert (len(conditions) == 1)
    tbl = list(conditions.keys())[0]
    conds = conditions[tbl]
    # logger.info("conds: %s", conds)
    if len(conds) == 1:

        cond = conds[0]
        # no selection, simple cardinality, return n
        # [SingleTablePushedDownCondition[badges]--join_keys[badges.Id]--non_key[None]--condition[None, None]--to_join[{}]]]
        if cond.non_key is None:
            return models[tbl].size

        # one selection
        if use_column_model:
            # logger.info("models[tbl].pdfs %s", models[tbl].pdfs.keys())
            model: Column = models[tbl].pdfs[cond.non_key.split(".")[1]]
            # logger.info("model is %s", model)
            domain = Domain(model.min, model.max)
            domain_query = Domain(
                cond.non_key_condition[0]-0.5, cond.non_key_condition[1]+0.5)
            domain.merge_domain(domain_query)

            grid_x, width = np.linspace(
                domain.min, domain.max, grid_size_x, retstep=True)
            pred = selectivity_array_single_column(model, grid_x, width)
            return np.sum(pred)*width*model.size
        else:
            model: Column2d = models[tbl].correlations["Id"][cond.non_key.split(".")[
                1]]
            jk_domain = [model.min[0]-0.5, model.max[0]+0.5]
            nk_domain = Domain(model.min[1]-0.5, model.max[1]+0.5)
            nk_domain_query = Domain(
                cond.non_key_condition[0]-0.5, cond.non_key_condition[1]+0.5)
            nk_domain.merge_domain(nk_domain_query)
            # logger.info("key domain %s", jk_domain)
            # logger.info("non_key domain %s", nk_domain)

            grid_x, width_x = np.linspace(
                *jk_domain, grid_size_x, retstep=True)
            grid_y, width_y = np.linspace(
                nk_domain.min, nk_domain.max, grid_size_y, retstep=True)
            # logger.info("widthx is  %s", width_x)
            # logger.info("widthy is  %s", width_y)

            if use_cdf:
                pred = model.pdf.predict_grid_with_y_range(grid_x,nk_domain)
            else:
                pred = selectivity_array_two_columns(
                    model, grid_x, grid_y, width_x, width_y)
            logger.warning("sum is %s",np.sum(pred))
            return np.sum(pred)*width_x*model.size

    elif len(conds) > 1:
        # tbl = list(conditions.keys())[0]
        cond0 = conds[0]
        jk = cond0.join_keys[0].split(".")[1]
        # jk = None
        # if 'Id' not in models[tbl].pdfs.keys():
        #     for jkk in models[tbl].pdfs.keys():
        #         if 'Id' in jkk:
        #             jk = jkk
        #             break
        # logger.info("keys is %s", models[tbl].pdfs.keys())
        jk_model = models[tbl].pdfs[jk]
        jk_domain = [jk_model.min, jk_model.max]
        # logger.info("x range is %s", jk_domain)
        grid_x, width_x = np.linspace(
            *jk_domain, grid_size_x, retstep=True)
        pred_x = selectivity_array_single_column(jk_model, grid_x, width_x)
        pred_xy = None
        for cond in conds:
            # logger.info("cond is %s", cond)
            assert (cond.non_key is not None)
            n_key = cond.non_key.split(".")[1]

            # logger.info("n_key is %s", models[tbl].correlations[jk])
            # logger.info("n_key is %s", models[tbl].correlations[jk])
            model2d: Column2d = models[tbl].correlations[jk][n_key]

            nk_domain = Domain(model2d.min[1], model2d.max[1])
            nk_domain_query = Domain(
                cond.non_key_condition[0]-0.5, cond.non_key_condition[1]+0.5)
            nk_domain.merge_domain(nk_domain_query)
            # nk_domain = [-0.5, 0.5]

            grid_y, width_y = np.linspace(
                nk_domain.min, nk_domain.max, grid_size_y, retstep=True)
            if pred_xy is None:
                if use_cdf:
                    # logger.warning("grid x is %s",grid_x)
                    logger.warning("domain is %s,%s",nk_domain.min,nk_domain.max)
                    pred_xy = model2d.pdf.predict_grid_with_y_range(grid_x,nk_domain)
                else:
                    pred_xy = selectivity_array_two_columns(
                        model2d, grid_x, grid_y, width_x, width_y)
                # logger.warning("pxy is %s",pred_xy)
                logger.warning("sum is %s",np.sum(pred_xy))
                pred_xy = np.divide(pred_xy, pred_x, out=np.zeros_like(
                    pred_xy), where=pred_x != 0)
                # logger.info("111max, min and  average are %s, %s, %s ",
                #             np.max(pred_xy), np.min(pred_xy), np.average(pred_xy))
                # logger.info("y range is %s", nk_domain)
                # logger.info("width is  %s", width_x)
                # logger.info("temp p1 is %s", np.sum(pred_xy)*width_x)
                # pred_xy = np.divide(pred_xy, pred_x)
                # logger.info("2max, min and  average are %s, %s, %s ",
                #             np.max(pred_xy), np.min(pred_xy), np.average(pred_xy))
            else:
                if use_cdf:
                    pred = model2d.pdf.predict_grid_with_y_range(grid_x,nk_domain)
                else:
                    pred = selectivity_array_two_columns(
                        model2d, grid_x, grid_y, width_x, width_y)
                    # logger.info("temp p3 is %s", np.sum(pred)*width_x)
                pred = np.divide(pred, pred_x, out=np.zeros_like(
                    pred), where=pred_x != 0)
                pred_xy = combine_selectivity_array(pred_xy, pred)
                # logger.info("final shape is %s", len(pred_xy))
                # logger.info("final shape is %s", pred_xy[0])
        pred_xy = combine_selectivity_array(pred_xy, pred_x)
        return np.sum(pred_xy)*width_x*jk_model.size
    return


def process_single_table_push_down_condition_for_join(tbl: str, models: dict[str, TableContainer], conditions: dict[str, list[SingleTablePushedDownCondition]], join_keys_grid: JoinKeysGrid, grid_size_y, use_column_model=True,use_cdf=False):
    logger.info("Table %s with conditions %s ", tbl, conditions)
    # tbl = list(conditions.keys())[0]
    cond0 = conditions[0]
    jk = cond0.join_keys[0].split(".")[1]

    jk_model = models[tbl].pdfs[jk]
    jk_grid = join_keys_grid.get_join_key_grid_for_table_jk(cond0.join_keys[0])
    # jk_domain = [jk_model.min, jk_model.max]
    # logger.info("x range is %s", jk_domain)
    grid_x, width_x = jk_grid.grid, jk_grid.width
    pred_x = selectivity_array_single_column(jk_model, grid_x, width_x)
    pred_xy = None
    for cond in conditions:
        # logger.info("cond is %s", cond)
        # assert (cond.non_key is not None)

        # joined table without selections.
        if cond.non_key is None:
            assert len(conditions) == 1
            mdl: Column = models[tbl].pdfs[jk]
            return mdl.pdf.predict(grid_x), width_x

        n_key = cond.non_key.split(".")[1]

        # logger.info("n_key is %s", models[tbl].correlations[jk])
        # logger.info("n_key is %s", models[tbl].correlations[jk])
        model2d: Column2d = models[tbl].correlations[jk][n_key]

        nk_domain = Domain(model2d.min[1], model2d.max[1])
        nk_domain_query = Domain(
            cond.non_key_condition[0]-0.5, cond.non_key_condition[1]+0.5)
        nk_domain.merge_domain(nk_domain_query)
        logger.info("domain is [%s,%s]", nk_domain.min, nk_domain.max)
        # nk_domain = [-0.5, 0.5]

        grid_y, width_y = np.linspace(
            nk_domain.min, nk_domain.max, grid_size_y, retstep=True)
        if pred_xy is None:
            if use_cdf:
                pred_xy = model2d.pdf.predict_grid_with_y_range(grid_x,nk_domain)
            else:
                pred_xy = selectivity_array_two_columns(
                    model2d, grid_x, grid_y, width_x, width_y)
            pred_xy = np.divide(pred_xy, pred_x, out=np.zeros_like(
                    pred_xy), where=pred_x != 0)
            # logger.info("111max, min and  average are %s, %s, %s ",
            #             np.max(pred_xy), np.min(pred_xy), np.average(pred_xy))
            # logger.info("y range is %s", nk_domain)
            # logger.info("width is  %s", width_x)
            # logger.info("temp p1 is %s", np.sum(pred_xy)*width_x)
            # pred_xy = np.divide(pred_xy, pred_x)
            # logger.info("2max, min and  average are %s, %s, %s ",
            #             np.max(pred_xy), np.min(pred_xy), np.average(pred_xy))
        else:
            if use_cdf:
                pred = model2d.pdf.predict_grid_with_y_range(grid_x,nk_domain)
            else:
                pred = selectivity_array_two_columns(
                    model2d, grid_x, grid_y, width_x, width_y)
                # logger.info("temp p3 is %s", np.sum(pred)*width_x)
            pred = np.divide(pred, pred_x, out=np.zeros_like(
                    pred), where=pred_x != 0)
            pred_xy = combine_selectivity_array(pred_xy, pred)
            # logger.info("final shape is %s", len(pred_xy))
            # logger.info("final shape is %s", pred_xy[0])
    pred_xy = combine_selectivity_array(pred_xy, pred_x)
    # return np.sum(pred_xy)*width_x*jk_model.size
    return pred_xy, width_x


def get_cartesian_cardinality(counters, tables_all):
    tables = list(tables_all.values())
    n = 1
    for tbl in tables:
        n *= counters[tbl]
    return n


if __name__ == '__main__':
    b = np.array([
        [1, 2],
        [3, 4],
        [5, 6],])
    a = np.array([1, 2])
    c = np.array([1, 2, 3])
    res = combine_selectivity_grid_with_two_arrays(a, b, c)
    minused = res - np.array([[1,  4],
                              [6, 16],
                              [15, 36]])
    assert (minused.all() == 0)

    res = array_multiply_grid(a, b)
    minused = res - np.array([9,  24])
    assert (minused.all() == 0)

    res = grid_multiply_array(b, c)
    minused = res - np.array([22, 28])
    assert (minused.all() == 0)
