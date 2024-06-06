# this is an example for p(yz|x)= p(y|x)p(z|x), assuming y and z are conditionally independent given x
# https://stats.stackexchange.com/questions/29510/proper-way-to-combine-conditional-probability-distributions-of-the-same-random-v
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
from joins.join_graph import get_join_hyper_graph
from joins.parser import get_max_dim, get_two_chained_query, parse_query_simple
from joins.plots import plot_line, plot_vec_sel_array
from joins.schema_base import identify_conditions, identify_key_values
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

    def query(self, query_str):
        logger.info("QUERY [%s]", query_str)
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
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
            self.models, conditions, join_cond, join_paths
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
    return models[tbl].size*selectivity


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
    return pathss


def multi_query_with_same_column(models, conditions, join_cond, join_paths):
    splits = join_paths[0][0].split(".")
    tbl = splits[0]
    jk = splits[1]

    hist = models[tbl].key_hist[jk].pdf
    for table_join_key in join_paths[0][1:-1]:
        splits1 = table_join_key.split(".")
        tbl1 = splits1[0]
        jk1 = splits1[1]
        hist1 = models[tbl1].key_hist[jk1].pdf
        hist = hist.join(hist1, update_statistics=True)

    splits1 = join_paths[0][-1].split(".")
    tbl1 = splits1[0]
    jk1 = splits1[1]
    hist1 = models[tbl1].key_hist[jk1].pdf
    res = hist.join(hist1)

    return np.sum(res)


def vec_sel_single_table_query(
    models: dict[str, TableContainer],
    conditions: list[SingleTablePushedDownCondition],
    grid_size_x=None,
    use_column_model=False,
    join_keys_grid=None,
    force_return_vec_sel_key=None,
    return_with_width_multiplied=True,
    return_width=False,
    bug_support_for_single_no_selection_join=False,
):
    assert len(conditions) == 1
    tbl = list(conditions.keys())[0]
    conds = conditions[tbl]

    # sz_min = np.Infinity
    # if tbl == "votes":
    #     logger.warning("hahha")
    # logger.warning("conds %s", conds)
    # if tbl == "users":
    #     logger.warning("!!!!!this is table users")

    # logger.info("conds: %s", conds)
    if len(conds) == 1:
        cond = conds[0]

        # no selection, simple cardinality, return n
        # [SingleTablePushedDownCondition[badges]--join_keys[badges.Id]--non_key[None]--condition[None, None]--to_join[{}]]]
        if cond.non_key is None:
            # logger.info(
            #     "table is %s: force_return_vec_sel_key %s",
            #     tbl,
            #     force_return_vec_sel_key,
            # )
            if force_return_vec_sel_key:
                # logger.info(
                #     "!!!~~~!!!!~~~~table [%s] with join key [%s]",
                #     tbl,
                #     force_return_vec_sel_key,
                # )
                # model: Column2d = (
                #     models[tbl].correlations_cdf["Id"][cond.non_key.split(".")[
                #         1]]
                #     if not force_return_vec_sel_key
                #     else models[tbl].correlations_cdf[force_return_vec_sel_key][
                #         'CreationDate'
                #     ]
                # )
                # jk_domain = [model.min[0], model.max[0]]
                # nk_domain = Domain(model.min[1], model.max[1], True, True)
                # # nk_domain_query = cond.non_key_condition
                # logger.info("jk_domain %s", jk_domain)
                # # logger.info("nk_domain_query %s", nk_domain_query)
                # # if nk_domain_query:
                # #     nk_domain.merge_domain(nk_domain_query)

                # if join_keys_grid:
                #     grid_i = join_keys_grid.get_join_key_grid_for_table_jk(
                #         tbl + "." + force_return_vec_sel_key
                #     )
                #     grid_x = grid_i.grid
                #     width_x = grid_i.width
                #     # logger.info("grid_x is %s", grid_x)
                # else:
                #     grid_x, width_x = np.linspace(
                #         *jk_domain, grid_size_x, retstep=True
                #     )

                # pred = model.pdf.predict_grid_with_y_range(grid_x, nk_domain)
                # if bug_support_for_single_no_selection_join:
                #     logger.info("bug support !~~~~~~~~~~~~~~~~~!!")
                #     return pred*width_x, width_x
                model: Column = models[tbl].pdfs[force_return_vec_sel_key]
                # if tbl == "votes":
                #     model: Column = models[tbl].pdfs["Id"]
                # logger.info("join_keys_grid %s", join_keys_grid.join_keys_grid)
                grid = join_keys_grid.get_join_key_grid_for_table_jk(
                    tbl + "." + force_return_vec_sel_key
                )
                # logger.info("model is %s",model)
                # logger.info("width is %s",join_keys_grid.join_keys_grid[0].width)
                # logger.info("grid is %s",join_keys_grid.join_keys_grid[0].grid)
                # logger.info(
                #     "table [%s] with selectivity is %s",
                #     tbl,
                #     np.sum(model.pdf.predict(grid.grid)) * grid.width,
                # )
                if return_width:
                    return model.pdf.predict(grid.grid) * grid.width, grid.width
                return model.pdf.predict(grid.grid) * grid.width

                # if bug_support_for_single_no_selection_join:
                #     return model.pdf.predict(grid.grid)*grid.width, grid.width
                # if return_with_width_multiplied:
                #     if return_width:
                #         return model.pdf.predict(grid.grid), grid.width
                #     return model.pdf.predict(grid.grid)
                # else:
                #     if return_width:
                #         return model.pdf.predict(grid.grid) / grid.width, grid.width
                #     return model.pdf.predict(grid.grid) / grid.width
            # sz_min = models[tbl].size
            return np.array([1.0])  # [models[tbl].size]

        # one selection
        if use_column_model:
            # logger.info("models[tbl].pdfs %s", models[tbl].pdfs.keys())
            model: Column = (
                models[tbl].cdfs[cond.non_key.split(".")[1]]
                if models[tbl].use_cdf
                else models[tbl].pdfs[cond.non_key.split(".")[1]]
            )
            # logger.info("model is %s", model)
            domain = Domain(model.min, model.max, True, True)
            domain_query = cond.non_key_condition
            domain.merge_domain(domain_query)
            pred = model.pdf.predict_domain(domain)
            # logger.debug("selectivity is %s", np.sum(pred))
            return pred
        # for cases if column model is not used.

        # TODO this is commented out !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # model = None
        # if force_return_vec_sel_key:
        # # logger.info("join key is %s", force_return_vec_sel_key)
        # # logger.info("join_keys_grid.join_keys_grid %s",
        # #             join_keys_grid.join_keys_grid)
        # model: Column = models[tbl].pdfs[force_return_vec_sel_key]
        # # if tbl == "votes":
        # #     model: Column = models[tbl].pdfs["Id"]
        # grid = join_keys_grid.get_join_key_grid_for_table_jk(
        #     tbl+"."+force_return_vec_sel_key)
        # return (
        #     model.pdf.predict(grid.grid)
        #     * grid.width
        # )
        # if tbl == "users":
        #     logger.warning("!!!!!this is table users %s", force_return_vec_sel_key)
        # logger.info(
        #     "!!!~~~!!!!~~~~table [%s] with join key [%s]", tbl, force_return_vec_sel_key
        # )
        model: Column2d = (
            models[tbl].correlations_cdf["Id"][cond.non_key.split(".")[1]]
            if not force_return_vec_sel_key
            else models[tbl].correlations_cdf[force_return_vec_sel_key][
                cond.non_key.split(".")[1]
            ]
        )
        jk_domain = [model.min[0], model.max[0]]
        nk_domain = Domain(model.min[1], model.max[1], True, True)
        nk_domain_query = cond.non_key_condition
        # logger.info("jk_domain %s", jk_domain)
        # logger.info("nk_domain_query %s", nk_domain_query)
        if nk_domain_query:
            nk_domain.merge_domain(nk_domain_query)
        # logger.info("nk_domain %s", nk_domain)

        # logger.info("join_keys_grid is %s", join_keys_grid.join_keys_grid)
        if join_keys_grid:
            grid_i = join_keys_grid.get_join_key_grid_for_table_jk(
                tbl + "." + force_return_vec_sel_key
            )
            grid_x = grid_i.grid
            width_x = grid_i.width
            # logger.info("grid_x is %s", grid_x)
        else:
            grid_x, width_x = np.linspace(
                *jk_domain, grid_size_x, retstep=True
            )  # TODO  done !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # grid_y, width_y = np.linspace(
        #     nk_domain.min, nk_domain.max, grid_size_y, retstep=True)
        # logger.info("grid x is %s", grid_x)
        # logger.warning("--------[single table query for table (%s)]-------------", tbl)
        pred = model.pdf.predict_grid_with_y_range(grid_x, nk_domain)
        # logger.info("pred is %s", pred[:2])
        # logger.info("sum is %s", np.sum(pred))
        # logger.info("sums x width is %s", np.sum(pred) * width_x)
        if bug_support_for_single_no_selection_join:
            # logger.info("bug support !~~~~~~~~~~~~~~~~~!!")
            return pred * width_x, width_x
        if return_with_width_multiplied:
            if return_width:
                return pred * width_x, width_x
            return pred * width_x  # , model.size
        else:
            if return_width:
                return pred, width_x
            return pred
    # multiple selection
    # logger.info("!" * 200)
    cond0 = conds[0]
    jk = cond0.join_keys[0].split(".")[1]
    jk_model = models[tbl].pdfs[jk]
    jk_domain = [jk_model.min, jk_model.max]
    # logger.info("x range is %s", jk_domain)
    if join_keys_grid:
        grid_i = join_keys_grid.get_join_key_grid_for_table_jk(tbl + "." + jk)
        grid_x = grid_i.grid
        width_x = grid_i.width
    else:
        grid_x, width_x = np.linspace(*jk_domain, grid_size_x, retstep=True)
    pred_x = vec_sel_single_column(jk_model, grid_x)

    pred_xys = []

    for condi in conds:
        if not join_keys_grid:
            sel = vec_sel_single_table_query(
                models, {tbl: [condi]}, grid_size_x, use_column_model=False
            )
        else:
            sel = vec_sel_single_table_query(
                models,
                {tbl: [condi]},
                use_column_model=False,
                join_keys_grid=join_keys_grid,
                force_return_vec_sel_key=force_return_vec_sel_key,
            )
        pred_xys.append(sel)
        # logger.debug("sub selectivity is %s", np.sum(sel))

        # logger.info("size is %s", sz)
        # if sz < sz_min:
        #     sz_min = sz

    # logger.info("predx is %s", np.sum(pred_x))
    pred = np.ones_like(pred_x)
    for pred_xyi in pred_xys:
        pred = vec_sel_multiply(
            pred, vec_sel_divide(pred_xyi, pred_x)) / width_x

    # logger.info("width x is %s", width_x)
    res = vec_sel_multiply(pred, pred_x)

    if bug_support_for_single_no_selection_join:
        return res * width_x, width_x
    if return_with_width_multiplied:
        if return_width:
            return res * width_x, width_x
        return res * width_x
    else:
        if return_width:
            return res, width_x
        return res
    # return res  # , sz_min


def vec_sel_multi_table_query(
    models,
    conditions,
    join_cond,
    join_keys_grid: JoinKeysGrid,
    grid_size_x=2000,
    grid_size_y=1000,
    join_keys_grid_1: JoinKeysGrid = None,
    return_with_width_multiplied=True,
    return_width=False,
):
    # logger.info("conditions: %s", conditions)
    # logger.info("join_keys_grid.join_keys_lists: %s", join_keys_grid.join_keys_lists)
    ps = {}
    widths = {}
    for tbl in conditions:
        predictions_within_table = []

        jk_id = None
        for jk_item in join_keys_grid.join_keys_lists[0]:
            # logger.info("jk_item %s", jk_item)
            if tbl in jk_item:
                jk_id = jk_item.split(".")[1]
                break
        # not found, means the second join path is used.
        if jk_id is None:
            for jk_item in join_keys_grid.join_keys_lists[1]:
                # logger.info("jk_item %s", jk_item)
                if tbl in jk_item:
                    jk_id = jk_item.split(".")[1]
                    break
        # logger.info("table with id %s, %s", tbl, jk_id)
        pred_p, width = vec_sel_single_table_query(
            models,
            {tbl: conditions[tbl]},
            join_keys_grid=join_keys_grid,
            force_return_vec_sel_key=jk_id,
            return_with_width_multiplied=True,
            return_width=True,
            bug_support_for_single_no_selection_join=True,
        )
        # logger.debug("[table %s with selectivity: %s", tbl, np.sum(pred_p))

        ps[tbl] = pred_p
    if join_keys_grid_1:
        predss = vec_sel_join(
            ps,
            join_cond,
            join_keys_grid_1,
            return_with_width_multiplied=False,
        )
    else:
        predss = vec_sel_join(
            ps,
            join_cond,
            join_keys_grid,
            return_with_width_multiplied=False,
        )
    if return_with_width_multiplied:
        if return_width:
            return predss * width, width
        return predss * width
    if return_width:
        return predss, width
    return predss


def vec_sel_single_column(column, x_grid):
    return column.pdf.predict(x_grid)


def vec_sel_multiply(sel1, sel2):
    return np.multiply(sel1, sel2)


def vec_sel_divide(sel1, sel2):
    # for 0 division, force to zero
    return np.divide(sel1, sel2, out=np.zeros_like(sel1), where=sel2 != 0)


def vec_sel_join(
    ps, join_cond, join_keys_grid, return_with_width_multiplied=True, return_width=False
):
    # logger.info("join_cond %s",join_cond)
    # logger.info("ps %s", ps)
    # logger.info("join_keys_grid %s", join_keys_grid.join_keys_grid)

    tbls = ps.keys()
    tbl0 = list(tbls)[0]
    width = join_keys_grid.join_keys_grid[0].width

    pred = ps[tbl0]  # /width
    # plot_line(pred)
    # logger.info("ssub of %s is %s", tbl0, np.sum(pred))
    for tbl in ps:
        if tbl != tbl0:
            # plot_line(ps[tbl])
            pred = vec_sel_multiply(pred, ps[tbl])
            # plot_line(pred)
            # logger.info("sub of %s is %s", tbl, np.sum(ps[tbl]))
    # logger.info("total selectivity is %s", np.sum(pred))
    if return_with_width_multiplied:
        if return_width:
            return pred * width, width
        return pred * width
    if return_width:
        return pred, width
    return pred


def get_cartesian_cardinality(counters, tables_all):
    tables = list(tables_all.values())
    n = 1
    for tbl in tables:
        n *= counters[tbl]
    return n


def vec_sel_multi_table_query_with_same_column(
    models,
    conditions,
    join_cond,
    join_keys_grid: JoinKeysGrid,
    join_keys_grid_1: JoinKeysGrid = None,
    return_with_width_multiplied=True,
    return_width=False,
):
    # logger.info("conditions: %s", conditions)
    # logger.info("join_keys_grid.join_keys_lists: %s", join_keys_grid.join_keys_lists)
    ps = {}
    widths = {}
    for tbl in conditions:
        predictions_within_table = []

        jk_id = None
        for jk_item in join_keys_grid.join_keys_lists[0]:
            # logger.info("jk_item %s", jk_item)
            if tbl in jk_item:
                jk_id = jk_item.split(".")[1]
                break
        # not found, means the second join path is used.
        if jk_id is None:
            for jk_item in join_keys_grid.join_keys_lists[1]:
                # logger.info("jk_item %s", jk_item)
                if tbl in jk_item:
                    jk_id = jk_item.split(".")[1]
                    break
        # logger.info("table with id %s, %s", tbl, jk_id)
        pred_p, width = vec_sel_single_table_query(
            models,
            {tbl: conditions[tbl]},
            join_keys_grid=join_keys_grid,
            force_return_vec_sel_key=jk_id,
            return_with_width_multiplied=False,
            return_width=True,
            bug_support_for_single_no_selection_join=True,
        )
        # logger.debug("[table %s with selectivities: %s", tbl, np.sum(pred_p))
        # logger.debug("[table %s 's selectivities: %s", tbl, pred_p[:2])

        ps[tbl] = pred_p

    if join_keys_grid_1:
        # logger.info("grid1")
        predss = vec_sel_join(
            ps,
            join_cond,
            join_keys_grid_1,
            return_with_width_multiplied=False,
        )
    else:
        # logger.info("grid")
        predss = vec_sel_join(
            ps,
            join_cond,
            join_keys_grid,
            return_with_width_multiplied=False,
        )

    # logger.info("shape is %s, %s", len(ps), len(ps[list(ps.keys())[0]]))
    # for p in ps:
    #     logger.info("sum is %s, details: %s", np.sum(ps[p]), ps[p][:5])
    # logger.info("final selectivity is %s", np.sum(predss))
    # logger.info("len of selectivity is %s", len(predss))
    # logger.info("predss is  %s", predss[:5])
    # plot_vec_sel_array(ps, predss)
    # if return_with_width_multiplied:
    #     if return_width:
    #         return predss * width, width
    #     return predss * width
    if return_width:
        return predss, width
    return predss


def count_of_vec_sel_multi_table_query_with_same_column(
    conditions,
):
    n_no_selection = 0
    n_with_selection = len(conditions)
    # logger.info("conditions: %s", conditions)

    for tbl in conditions:
        conditions_in_table = conditions[tbl]
        if len(conditions_in_table) == 1:
            cond = conditions_in_table[0]
            if cond.non_key is None:
                n_no_selection += 1
                n_with_selection -= 1
    return n_no_selection, n_with_selection
