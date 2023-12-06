import numpy as np

from joins.base_logger import logger

# from joins.table import TableContainer


class Domain:
    def __init__(self, mins=-np.Infinity, maxs=np.Infinity, left=False, right=True, is_categorical=True) -> None:
        self.min = mins
        self.max = maxs
        self.left = left
        self.right = right
        self.is_categorical = is_categorical

    def merge_domain(self, d1):
        # print("d1: ", d1)
        # print("type is ", type(d1))
        if d1.min > self.min:
            self.min = d1.min
            self.left = d1.left
        if d1.max < self.max:
            self.max = d1.max
            self.right = d1.right

    def __str__(self) -> str:
        low = ""
        if self.left:
            low += "["
        else:
            low += "("

        high = ""
        if self.right:
            high += "]"
        else:
            high += ")"

        return f"{low}{self.min}, {self.max}{high}"


class JoinKeyGrid:
    def __init__(self, low, high, grid_size) -> None:
        self.grid, self.width = np.linspace(low, high, grid_size, retstep=True)


class JoinKeysGrid:
    def __init__(self) -> None:
        self.join_keys_lists = []
        self.join_keys_domain = []
        self.join_keys_grid: list[JoinKeyGrid] = []

    def calculate_push_down_join_keys_domain(self, conditions, join_cond, models: dict, tabls_all, grid_size):
        # note, selection on join key is not supported yet.
        if join_cond is None:
            # tbl = list(tabls_all.values())[0]
            return None, None
        # assert (join_cond is not None)
        # if join_cond is None:
        #     k = list(ps.keys())[0]
        #     return np.sum(ps[k])
        join_keys = []
        join_keys_domain = []
        join_cond_copy = join_cond.copy()
        # logger.info("in merge_predictions")
        # logger.info("len is %s", len(ps))
        while len(join_cond_copy) > 0:
            join_cond = join_cond_copy.pop()
            tk1, tk2 = join_cond.replace(" ", "").split("=")
            t1, k1 = tk1.split(".")
            t2, k2 = tk2.split(".")
            domain1 = Domain(models[t1].pdfs[k1].min, models[t1].pdfs[k1].max)
            domain2 = Domain(models[t2].pdfs[k2].min, models[t2].pdfs[k2].max)
            # merged = merge_domain(domain1, domain2)
            domain1.merge_domain(domain2)

            # check existence and update domain
            idx1 = get_idx_in_lists(tk1, join_keys)
            idx2 = get_idx_in_lists(tk2, join_keys)
            if idx1 == -1 and idx2 == -1:
                join_key_pair = [tk1, tk2]
                join_keys.append(join_key_pair)
                join_keys_domain.append(domain1)
            elif idx1 >= 0:
                join_keys[idx1].append(tk2)
                join_keys_domain[idx1].merge_domain(domain1)
            elif idx2 >= 0:
                join_keys[idx2].append(tk1)
                join_keys_domain[idx2].merge_domain(domain1)
            else:
                logger.error(
                    "unexpected behavior as the join condition appear twice")
        self.join_keys_lists = join_keys
        self.join_keys_domain = join_keys_domain

        for domain in join_keys_domain:
            grid = JoinKeyGrid(domain.min, domain.max, grid_size)
            self.join_keys_grid.append(grid)

    def get_join_key_grid_for_table_jk(self, jk) -> JoinKeyGrid:
        idx = get_idx_in_lists(jk, self.join_keys_lists)
        assert (idx >= 0)
        return self.join_keys_grid[idx]


def get_idx_in_lists(k, lists):
    for idx, ls in enumerate(lists):
        if k in ls:
            return idx
    return -1


# def merge_domain(d1: Domain, d2: Domain):
#     return [max(l1[0], l2[0]), min(l1[1], l2[1])]


class SingleTablePushedDownCondition:
    def __init__(self, tbl: str, join_keys: list[str], non_key: str, non_key_condition: dict[str, dict], to_join, key_conditions=None) -> None:
        self.tbl = tbl
        # currently only support at most 2 join keys in a single table
        assert (len(join_keys) <= 2)
        self.join_keys = join_keys
        self.non_key = non_key
        self.non_key_condition = non_key_condition
        # selection on join key is not supported, but could be easily supported.
        assert (key_conditions is None)

        self.to_join = to_join

    def __str__(self) -> str:
        join_str = str(self.to_join)

        return f"SingleTablePushedDownCondition[{self.tbl}]--join_keys[{','.join(self.join_keys)}]--non_key[{self.non_key}]--condition{self.non_key_condition }--to_join[{join_str}]]"

    def __repr__(self) -> str:
        return self.__str__()


def generate_push_down_conditions(tables_all, table_query, join_cond, join_keys):
    # logger.info("tables_all %s", tables_all)
    # logger.info("table_query %s", table_query)
    # logger.info("join_cond %s", join_cond)
    # logger.info("join_keys %s", join_keys)
    conditions = {}
    if len(tables_all) == 1:
        to_join = {}
        tbl = list(tables_all.values())[0]
        default_primary_key = [tbl+"."+"Id"]

        if table_query and tbl in table_query:
            # push down single table condition
            single_table_conditions = []
            for non_key in table_query[tbl]:
                condition = Domain(-np.Infinity, np.Infinity)
                for op in table_query[tbl][non_key]:
                    val = table_query[tbl][non_key][op]
                    if '<=' in op:
                        condition.max = val
                        condition.right = True
                    elif '<' in op:
                        condition.max = val
                        condition.right = False
                    elif '>=' in op:
                        condition.min = val
                        condition.left = True
                    elif '>' in op:
                        condition.min = val
                        condition.left = False
                    elif '==' in op or '=' in op:
                        condition = Domain(val, val, True, True)
                    else:
                        logger.error("unexpected operation")

                con = SingleTablePushedDownCondition(
                    tbl, default_primary_key, non_key, condition, to_join, None)
                single_table_conditions.append(con)
        else:
            single_table_conditions = []
            con = SingleTablePushedDownCondition(
                tbl, default_primary_key, None, None, to_join, None)
            single_table_conditions.append(con)
        conditions[tbl] = single_table_conditions
        return conditions

    for tbl in join_keys:
        # print(tbl)
        # print(join_keys[tbl])
        join_keyss = list(join_keys[tbl])
        # print("join_keyss", join_keyss)
        # prepare to_join condition
        to_join = {}
        for jk in join_keyss:
            # logger.info("jk is %s", jk)
            for join_condition in join_cond:
                if jk in join_condition:
                    to_j = join_condition.replace(
                        jk, "").replace("=", "").replace(" ", "")
                    # logger.info("to_j,%s", to_j)
                    to_tbl, to_k = to_j.split(".")
                    if to_tbl not in to_join:
                        to_join[to_tbl] = []
                    to_join[to_tbl].append(to_k)

        if tbl in table_query:
            # push down single table condition
            single_table_conditions = []
            for non_key in table_query[tbl]:
                condition = [-np.Infinity, np.Infinity]
                for op in table_query[tbl][non_key]:
                    val = table_query[tbl][non_key][op]
                    if '<=' in op or '<' in op:
                        condition[1] = val
                    elif '>=' in op or '>' in op:
                        condition[0] = val
                    elif '==' in op or '=' in op:
                        condition = [val-0.5, val+0.5]
                    else:
                        logger.error("unexpected operation")

                con = SingleTablePushedDownCondition(
                    tbl, join_keyss, non_key, condition, to_join, None)
                single_table_conditions.append(con)
        else:
            single_table_conditions = []
            con = SingleTablePushedDownCondition(
                tbl, join_keyss, None, None, to_join, None)
            single_table_conditions.append(con)
        conditions[tbl] = single_table_conditions
    return conditions
