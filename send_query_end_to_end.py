import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import psycopg2

from joins.tools import save_predictions_to_file, save_predictions_to_file3


def send_query(dataset, method_name, query_file, save_folder, iteration=None):
    conn = psycopg2.connect(
        # options="-c statement_timeout=600s",
        database=dataset,
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port=5432,
    )
    cursor = conn.cursor()

    with open(query_file, "r") as f:
        queries = f.readlines()  # [587:588]

    # cursor.execute('SET debug_card_est=true')
    # cursor.execute('SET print_sub_queries=true')
    # cursor.execute('SET print_single_tbl_queries=true')
    methods = [
        # "factorjoin",
        # "dhist",
        # "deepdb",
        # "flat",
        # "neurocard",
        # "wjsample",
        # "truth",
        # "postgres",
    ]
    # method = "postgres"  # "factorjoin", "dhist", "deepdb", "flat", "neurocard", "wjsample", "truth", "postgres"
    for method in methods:
        single = "stats_CEB_single_" + method
        multi = "stats_CEB_sub_queries_" + method

        cursor.execute("SET ml_cardest_enabled=false;")
        cursor.execute("SET query_no=0;")
        cursor.execute(f"SET ml_cardest_fname='{single}.txt';")
        cursor.execute("SET ml_joinest_enabled=true;")
        cursor.execute("SET join_est_no=0;")
        cursor.execute(f"SET ml_joinest_fname='{multi}.txt';")

        planning_time = []
        execution_time = []
        predictions = []
        ratios = []
        truths = []
        for no, query_str in enumerate(queries):
            if "||" in query_str:
                query = query_str.split("||")[1]
            print(f"Executing query {no}")

            try:
                start = time.time()
                cursor.execute("EXPLAIN ANALYZE " + query)
                res = cursor.fetchall()

                planning_time.append(
                    float(res[-2][0].split(":")[-1].split("ms")[0].strip())
                )
                execution_time.append(
                    float(res[-1][0].split(":")[-1].split("ms")[0].strip())
                )
                truths.append(1)
                end = time.time()
                print(f"{no}-th query finished in {end-start}")

            except psycopg2.errors.QueryCanceled:
                truths.append(-1)
                execution_time.append(0)
                planning_time.append(0)
                print(f"{no}-th query timeout!")
                conn.rollback()

        save_predictions_to_file3(
            truths,
            execution_time,
            planning_time,
            "truth",
            "execution-time",
            "plan-time",
            "results/stats/end_to_end/" + method + ".csv",
        )
    cursor.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="stats", help="Which dataset to be used")
    parser.add_argument(
        "--method_name",
        default="workloads/stats_CEB/estimates/truth.txt",
        help="save estimates",
    )
    parser.add_argument(
        "--query_file",
        default="workloads/stats_CEB/stats_CEB.sql",
        help="Query file location",
    )
    parser.add_argument(
        "--with_true_card",
        action="store_true",
        help="Is true cardinality included in the query?",
    )
    parser.add_argument(
        "--save_folder",
        default="workloads/stats_CEB/estimates/",
        help="Query file location",
    )
    parser.add_argument(
        "--iteration", type=int, default=None, help="Number of iteration to read"
    )

    args = parser.parse_args()

    if args.iteration:
        for i in range(args.iteration):
            send_query(
                args.dataset, args.method_name, args.query_file, args.save_folder, i + 1
            )
    else:
        send_query(args.dataset, args.method_name, args.query_file, args.save_folder)
