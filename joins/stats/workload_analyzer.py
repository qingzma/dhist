from argparse import ArgumentParser

from joins.args import parse_args
from joins.base_logger import logger
from joins.parser import parse_single_table_query


def analyze_stats(args: ArgumentParser):
    tbl_columns = {}
    with open(args.query, 'r', encoding="utf-8") as f:
        queries = f.readlines()

    for query_str in queries:  # [:10]
        query = query_str.split("||")[0][:-1]
        tbl_name, cols = parse_single_table_query(query)

        if cols is not None:
            if tbl_name not in tbl_columns:
                tbl_columns[tbl_name] = set([cols])
            else:
                tbl_columns[tbl_name].add(cols)

    for tbl in tbl_columns:
        tbl_columns[tbl] = list(tbl_columns[tbl])

    logger.info("tbl_columns %s", tbl_columns)


if __name__ == '__main__':
    arguments = ["--query",
                 "workloads/stats_CEB/sub_plan_queries/stats_CEB_single_table_sub_query.sql"]
    args = parse_args(arguments)
    analyze_stats(args)
