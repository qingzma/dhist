from argparse import ArgumentParser

from joins.args import parse_args
from joins.base_logger import logger
from joins.parser import parse_single_table_query
import pandas as pd


def analyze_job_light_single(args: ArgumentParser):
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


def analyze_job_light(args: ArgumentParser):
    tbl_columns = {}
    with open(args.query, 'r', encoding="utf-8") as f:
        queries = f.readlines()

    for query_str in queries:  # [:10]
        # query = query_str.split("||")[0][:-1]
        query = query_str.split('WHERE')[-1].split('AND')
        for q in query:
            if '=' not in q:
                continue
            left, right = q.split('=')
            if '.' in left and '.' in right:
                tbl_name_left = left.split('.')[0].strip()
                attr_left = left.split('.')[1].strip()
                tbl_name_right = right.split('.')[0].strip()
                attr_right = right.split('.')[1].strip()
                if tbl_name_left not in tbl_columns:
                    tbl_columns[tbl_name_left] = set([(attr_left)])
                else:
                    tbl_columns[tbl_name_left].add((attr_left))
                if tbl_name_right not in tbl_columns:
                    tbl_columns[tbl_name_right] = set([(attr_right)])
                else:
                    tbl_columns[tbl_name_right].add((attr_right))
    print(tbl_columns)


def get_low_high():
    tables = ['movie_companies', 'movie_info_idx', 'title', 'movie_keyword', 'cast_info', 'movie_info']
    for table in tables:
        path = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/imdb/imdb/{}.csv'.format(table)
        try:
            df = pd.read_csv(path, on_bad_lines='skip', sep=',')
        except pd.errors.ParserError as e:
            print(f"ParserError: {e}")
        # df = pd.read_csv(path, sep=',')
        if table == 'title':
            col = 0
        elif table == 'cast_info':
            col = 2
        else:
            col = 1
        fill_value = -1  # 你可以选择任何合适的值
        max_value = df.iloc[:, col].fillna(fill_value).max()
        min_value = df.iloc[:, col].fillna(fill_value).min()
        print(table, min_value, max_value)


if __name__ == '__main__':
    root_path = '/home/lrr/Documents/End-to-End-CardEst-Benchmark/workloads/job-light/sub_plan_queries/job_light_sub_query.sql'
    arguments = ["--query",
                 '/home/lrr/Documents/End-to-End-CardEst-Benchmark/workloads/job-light/sub_plan_queries/job_light_sub_query.sql']
    args = parse_args(arguments)
    # analyze_job_light(args)
    get_low_high()

# tbl_columns {'movie_companies': [('company_Id',), ('company_type_Id',)], 'movie_info_idx': [('info_type_Id',)], 'title': [('production_year',), ('kind_Id', 'production_year'), ('kind_Id',)], 'movie_keyword': [('keyword_Id',)], 'cast_info': [('role_Id',)], 'movie_info': [('info_type_Id',)]}