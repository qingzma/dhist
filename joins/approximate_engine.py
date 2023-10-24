import scipy.integrate as integrate

from joins.base_logger import logger
from joins.join_graph import get_join_hyper_graph
from joins.parser import parse_query_simple
from joins.schema_base import identify_conditions, identify_key_values


class ApproximateEngine:
    def __init__(self,  models=None, auto_grid=True) -> None:
        self.models: dict = models
        self.auto_grid = auto_grid
        self.all_keys, self.equivalent_keys, self.table_keys = identify_key_values(
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

        print(query_str)
        # print(tables_all)
        print("table_queries\n", table_queries)
        # print(join_cond)
        # print(join_keys)
        # print(equivalent_group)
        # exit()
        print("key_conditions\n", key_conditions)
        print("non_key_conditions\n", non_key_conditions)
        return key_conditions, non_key_conditions

    def integrate1d(self, model_name, l, h):
        if self.auto_grid:
            return integrate.quad(lambda x: self.models[model_name].predict(x), l, h)
        else:
            logger.warning("grid integral is not implemented yet")

    def integrate2d(self,):
        pass
