# this is an example for p(yz|x)= p(y|x)p(z|x), assuming y and z are conditionally independent given x
# https://stats.stackexchange.com/questions/29510/proper-way-to-combine-conditional-probability-distributions-of-the-same-random-v
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
from joins.schema_base import identify_conditions, identify_key_values
from joins.stats.schema import get_stats_relevant_attributes
from joins.table import Column, Column2d, TableContainer

from joins.plots import plot_line


class Engine:
    def __init__(
        self, models: dict[str, TableContainer] = None, auto_grid=True, use_cdf=True
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
        tables_all, table_query, join_cond, join_keys = parse_query_simple(query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys
        )
        max_dim = get_max_dim(join_keys)
        logger.info("join_cond is %s", join_cond)
        logger.info("tables_all is %s", tables_all)
        logger.info("table_query is %s", table_query)
        logger.info("join_keys is %s", join_keys)
        logger.info("conditions %s", conditions)
        logger.info("max_dim %s", max_dim)

        # single table query
        if len(tables_all) == 1:
            vec_sel = vec_sel_single_table_query(
                self.models, conditions, self.grid_size_x
            )
            tbl = list(conditions.keys())[0]
            n = self.models[tbl].size
            return np.sum(vec_sel) * n

        # join query

        # only a single join key is allowed in a table
        if max_dim == 2:
            target_tbl, join_paths = get_two_chained_query(join_keys, join_cond)
            predictions_in_paths = {}
            for jk in join_paths.keys():
                conds = {key: conditions[key] for key in join_paths[jk]}
                tbls = {}
                for k in tables_all:
                    if tables_all[k] in join_paths[jk]:
                        tbls[k] = tables_all[k]

                logger.info("_" * 120)
                logger.info("conds is %s", conds)
                logger.info("tbls is %s", tbls)

                join_keys_grid = JoinKeysGrid()
                join_keys_grid.calculate_push_down_join_keys_domain(
                    conds, join_cond, self.models, tbls, self.grid_size_x
                )
                if len(tbls) == 1:
                    k = conds[list(tbls.values())[0]][0].join_keys[0].split(".")[1]
                    # print("k is ", k)
                    pred = vec_sel_single_table_query(
                        self.models,
                        conds,
                        self.grid_size_x,
                        force_return_vec_sel_key=k,
                        join_keys_grid=join_keys_grid,
                    )
                    tbl = list(conds.keys())[0]
                    n = self.models[tbl].size
                    predictions_in_paths[jk] = pred
                else:
                    n = get_cartesian_cardinality(self.counters, tbls)
                    pred = vec_sel_multi_table_query(
                        self.models, conds, join_cond, join_keys_grid
                    )
                    predictions_in_paths[jk] = pred

                logger.info("predictions_in_paths is %s", predictions_in_paths)
                logger.info("n is %s", n)
                logger.info("*" * 200)

            return len(join_keys_grid.join_keys_domain)  # TODO support this
        elif max_dim > 2:
            logger.error("length is  [%i]", len(join_keys_grid.join_keys_domain))
            logger.error("is 3 [%s]", query_str)
            exit()

        # for cases with max_dim==1
        join_keys_grid = JoinKeysGrid()
        join_keys_grid.calculate_push_down_join_keys_domain(
            conditions, join_cond, self.models, tables_all, self.grid_size_x
        )
        logger.info("join_keys_grid %s", join_keys_grid.join_keys_domain)
        logger.info(
            "join_keys_grid.join_keys_domain %s", join_keys_grid.join_keys_domain
        )
        assert len(join_keys_grid.join_keys_domain) == 1
        # join_keys_lists, join_keys_domain = calculate_push_down_join_keys_domain(
        #     conditions, join_cond, self.models, tables_all)

        n = get_cartesian_cardinality(self.counters, tables_all)
        pred = vec_sel_multi_table_query(
            self.models, conditions, join_cond, join_keys_grid
        )

        # logger.info("cartesian is %E", n)
        # logger.info("pred is %s ", np.sum(pred))

        return np.sum(pred) * n


def vec_sel_single_table_query(
    models: dict[str, TableContainer],
    conditions: list[SingleTablePushedDownCondition],
    grid_size_x=None,
    use_column_model=False,
    join_keys_grid=None,
    force_return_vec_sel_key=None,
):
    assert len(conditions) == 1
    tbl = list(conditions.keys())[0]
    conds = conditions[tbl]

    # sz_min = np.Infinity

    logger.info("conds: %s", conds)
    if len(conds) == 1:
        cond = conds[0]

        # no selection, simple cardinality, return n
        # [SingleTablePushedDownCondition[badges]--join_keys[badges.Id]--non_key[None]--condition[None, None]--to_join[{}]]]
        if cond.non_key is None:
            if force_return_vec_sel_key:
                model: Column = models[tbl].pdfs[force_return_vec_sel_key]
                # logger.info("model is %s",model)
                # logger.info("width is %s",join_keys_grid.join_keys_grid[0].width)
                # logger.info("grid is %s",join_keys_grid.join_keys_grid[0].grid)
                return (
                    model.pdf.predict(join_keys_grid.join_keys_grid[0].grid)
                    * join_keys_grid.join_keys_grid[0].width
                )
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
            logger.info("selectivity is %s", np.sum(pred))
            return pred
        # for cases if column model is not used.
        model = None
        if force_return_vec_sel_key:
            # logger.info("join key is %s", force_return_vec_sel_key)
            model: Column = models[tbl].pdfs[force_return_vec_sel_key]
            return (
                model.pdf.predict(join_keys_grid.join_keys_grid[0].grid)
                * join_keys_grid.join_keys_grid[0].width
            )

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

        if join_keys_grid:
            grid_x = join_keys_grid.join_keys_grid[0].grid
            width_x = join_keys_grid.join_keys_grid[0].width
        else:
            grid_x, width_x = np.linspace(
                *jk_domain, grid_size_x, retstep=True
            )  # TODO  done !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # grid_y, width_y = np.linspace(
        #     nk_domain.min, nk_domain.max, grid_size_y, retstep=True)
        # logger.info("grid x is %s", grid_x)
        pred = model.pdf.predict_grid_with_y_range(grid_x, nk_domain)
        # logger.info("pred is %s", pred)
        # logger.info("sum is %s", np.sum(pred))
        # logger.info("sums is %s", np.sum(pred)*width_x)
        return pred * width_x  # , model.size
    # multiple selection
    cond0 = conds[0]
    jk = cond0.join_keys[0].split(".")[1]
    jk_model = models[tbl].pdfs[jk]
    jk_domain = [jk_model.min, jk_model.max]
    # logger.info("x range is %s", jk_domain)
    if join_keys_grid:
        grid_x = join_keys_grid.join_keys_grid[0].grid
        width_x = join_keys_grid.join_keys_grid[0].width
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
        pred = vec_sel_multiply(pred, vec_sel_divide(pred_xyi, pred_x)) / width_x

    # logger.info("width x is %s", width_x)
    res = width_x * vec_sel_multiply(pred, pred_x)
    return res  # , sz_min


def vec_sel_multi_table_query(
    models,
    conditions,
    join_cond,
    join_keys_grid: JoinKeysGrid,
    grid_size_x=2000,
    grid_size_y=1000,
):
    # logger.info("conditions: %s", conditions)
    ps = {}
    widths = {}
    for tbl in conditions:
        predictions_within_table = []

        jk_id = None
        for jk_item in join_keys_grid.join_keys_lists[0]:
            # logger.info("jk_item %s",jk_item)
            if tbl in jk_item:
                jk_id = jk_item.split(".")[1]
                break
        # logger.info("table with id %s, %s", tbl, jk_id)
        pred_p = vec_sel_single_table_query(
            models,
            {tbl: conditions[tbl]},
            join_keys_grid=join_keys_grid,
            force_return_vec_sel_key=jk_id,
        )
        logger.info("[table %s with selectivity: %s", tbl, np.sum(pred_p))

        ps[tbl] = pred_p
    predss = vec_sel_join(ps, join_cond, join_keys_grid)
    return predss


def vec_sel_single_column(column, x_grid):
    return column.pdf.predict(x_grid)


def vec_sel_multiply(sel1, sel2):
    return np.multiply(sel1, sel2)


def vec_sel_divide(sel1, sel2):
    # for 0 division, force to zero
    return np.divide(sel1, sel2, out=np.zeros_like(sel1), where=sel2 != 0)


def vec_sel_join(ps, join_cond, join_keys_grid):
    # logger.info("join_cond %s",join_cond)
    # logger.info("join_keys_grid %s",join_keys_grid.join_keys_grid)
    tbls = ps.keys()
    tbl0 = list(tbls)[0]
    width = join_keys_grid.join_keys_grid[0].width

    pred = ps[tbl0]
    # plot_line(pred)
    # logger.info("ssub of %s is %s", tbl0, np.sum(pred))
    for tbl in ps:
        if tbl != tbl0:
            # plot_line(ps[tbl])
            pred = vec_sel_multiply(pred, ps[tbl])
            # plot_line(pred)
            # logger.info("sub of %s is %s", tbl, np.sum(ps[tbl]))
    return pred / width


def get_cartesian_cardinality(counters, tables_all):
    tables = list(tables_all.values())
    n = 1
    for tbl in tables:
        n *= counters[tbl]
    return n
