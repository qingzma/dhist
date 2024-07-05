import os
import pickle
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from joins.approximate_engine import ApproximateEngine
from joins.base_logger import logger
from joins.engine_topk import EngineTopK as Engine
from joins.tools import save_predictions_to_file


def evaluate_stats(args: ArgumentParser):
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    with open(args.query, "r", encoding="utf-8") as f:
        queries = f.readlines()

    engine = Engine(models=model, use_cdf=args.cdf)

    latency = []
    pred = []
    qerror = []
    ratios = []

    bad_queries = []
    cnt_key = []
    cnt_non_key = []
    for query_str in queries:
        query = query_str.replace(";", "").replace("\n", "")
        if query == "":
            pred.append(-1)
            latency.append(0)
        else:
            query = query_str.split("||")[0][:-1]
            # true_card = int(query_str.split("||")[-1])
            # query = query_str.split("||")[-1][:-1]
            # true_card = int(query_str.split("||")[0])
            t = time.time()
            res = engine.query(query)
            # cnt_key.append(len(key_conditions))
            # cnt_non_key.append(len(non_key_conditions))
            # if res == 2.0 or true_card == 0:
            #     pred.append(-1)
            #     latency.append(-1)
            #     continue
            # if res / true_card <= 5e-18:
            #     continue
            pred.append(res)
            latency.append(time.time() - t)

        # qerror.append(max(res/true_card, true_card/res))
        # qerror.append(res / true_card)
        # ratios.append(res / true_card)
        # if (res/true_card > 10000):
        #     exit()
        # if res / true_card > 1e5:
        #     # if res / true_card > 1e5:
        #     bad_queries.append(query)
        #     logger.info("-" * 800)
        #     logger.info("true is %s, pred is %s", true_card, res)
        #     logger.info("query is %s", query)
        #     logger.info("-" * 800)
        # exit()
        # logger.info("qerror is %s", max(res/true_card, true_card/res))
    # logger.info("max is %s", max(cnt_key))
    # logger.info("max of non is %s", max(cnt_non_key))

    # qerror = np.asarray(qerror)
    # logger.info(f"qerror is {qerror}")
    # for i in [1, 50, 90, 95, 99, 100]:
    #     logger.info(f"q-error {i}% percentile is {np.percentile(qerror, i)}")

    # logger.info(f"pred is  {pred}")
    logger.info(f"average latency per query is {np.mean(latency)}")
    logger.info(f"total estimation time is {np.sum(latency)}")
    logger.info(f"number of queries is {len(pred)}")
    # logger.info("bad queries\n")
    # logger.info("-" * 100)
    # for q in bad_queries:
    #     logger.info(q)

    # save_predictions_to_file(
    #     pred,
    #     latency,
    #     "card",
    #     "card-time",
    #     # "workloads/stats_CEB/no_range_predicates/5.topk_100_40.csv",
    #     "results/stats/multiple_tables/updates/cardall.csv",
    # )

    # logbins = np.logspace(np.log10(min(ratios)), np.log10(max(ratios)), 31)
    # plt.xscale("log")
    # plt.hist(ratios, bins=logbins)
    # plt.show()
    # model_name = model["name"].replace(".pkl", "") + ".txt"

    # save_res = True
    # if save_res:
    #     with open(model_name, "w") as f:
    #         for p in pred:
    #             f.write(str(p) + "\n")
