import os
import pickle
import time
from argparse import ArgumentParser

import numpy as np

from joins.approximate_engine import ApproximateEngine
from joins.base_logger import logger


def evaluate_stats(args: ArgumentParser):
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    with open(args.query, 'r', encoding="utf-8") as f:
        queries = f.readlines()

    engine = ApproximateEngine(models=model)

    latency = []
    pred = []
    # qerror = []

    cnt_key = []
    cnt_non_key = []
    for query_str in queries[:1]:  # [:10]
        query = query_str.split("||")[0][:-1]
        true_card = int(query_str.split("||")[-1])
        t = time.time()
        key_conditions, non_key_conditions = engine.query(query)
        cnt_key.append(len(key_conditions))
        cnt_non_key.append(len(non_key_conditions))
        # pred.append(res)
        latency.append(time.time() - t)
        # qerror.append(max(res/true_card, true_card/res))
    logger.info("max is %s", max(cnt_key))
    logger.info("max of non is %s", max(cnt_non_key))

    # qerror = np.asarray(qerror)
    # for i in [50, 90, 95, 99, 100]:
    #     logger.info(f"q-error {i}% percentile is {np.percentile(qerror, i)}")

    logger.info(f"average latency per query is {np.mean(latency)}")
    logger.info(f"total estimation time is {np.sum(latency)}")

    model_name = model['name'].replace(".pkl", '')+'.txt'

    save_res = True
    if save_res:
        with open(model_name, "w") as f:
            for p in pred:
                f.write(str(p) + "\n")
