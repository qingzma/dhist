import os
import pickle
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from joins.approximate_engine import ApproximateEngine
from joins.engine import Engine
from joins.base_logger import logger


def evaluate_stats(args: ArgumentParser):
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    with open(args.query, 'r', encoding="utf-8") as f:
        queries = f.readlines()

    engine = Engine(models=model, use_cdf=args.cdf)

    latency = []
    pred = []
    qerror = []
    ratios = []

    cnt_key = []
    cnt_non_key = []
    for query_str in queries:  # [:10]
        query = query_str.split("||")[0][:-1]
        true_card = int(query_str.split("||")[-1])
        t = time.time()
        res = engine.query(
            query)
        # cnt_key.append(len(key_conditions))
        # cnt_non_key.append(len(non_key_conditions))
        pred.append(res)
        latency.append(time.time() - t)

        qerror.append(max(res/true_card, true_card/res))
        ratios.append(res/true_card)
        # if (res/true_card > 10000):
        #     exit()
        if max(res/true_card, true_card/res) > 1e20:
            logger.info("-"*80)
            logger.info("true is %s, pred is %s", true_card, pred)
            logger.info("-"*80)
            # exit()
        logger.info("qerror is %s", max(res/true_card, true_card/res))
    # logger.info("max is %s", max(cnt_key))
    # logger.info("max of non is %s", max(cnt_non_key))

    qerror = np.asarray(qerror)
    logger.info(f"qerror is {qerror}")
    for i in [50, 90, 95, 99, 100]:
        logger.info(f"q-error {i}% percentile is {np.percentile(qerror, i)}")

    logger.info(f"average latency per query is {np.mean(latency)}")
    logger.info(f"total estimation time is {np.sum(latency)}")

    logbins = np.logspace(np.log10(min(ratios)),
                          np.log10(max(ratios)), 100)
    plt.xscale('log')
    plt.hist(ratios, bins=logbins)
    plt.show()
    model_name = model['name'].replace(".pkl", '')+'.txt'

    save_res = True
    if save_res:
        with open(model_name, "w") as f:
            for p in pred:
                f.write(str(p) + "\n")
