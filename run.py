import argparse
import os
import sys
import time

from joins.args import parse_args
from joins.base_logger import logger
from joins.stats.evaluate_stats import evaluate_stats
from joins.stats.train_stats import train_stats
from joins.tools import convert_time_to_int

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    os.makedirs("logs", exist_ok=True)

    logger.info("Start running experiment for %s", args.dataset)

    if args.dataset == "stats":
        if args.preprocess:
            logger.info("start pre-processing the data")
            convert_time_to_int(args.data_folder)

        elif args.train:
            logger.info("start training models")
            start_time = time.time()
            train_stats(args)
            end_time = time.time()
            logger.info(
                "Training completed: total training time is %.6f s.",
                end_time - start_time,
            )

        elif args.evaluate:
            logger.info("start evaluating models")
            start_time = time.time()
            evaluate_stats(args)
            end_time = time.time()
            logger.info(
                "Evaluation completed: total evaluation time is %.6f s.",
                end_time - start_time,
            )
