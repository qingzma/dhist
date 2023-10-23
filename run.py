import argparse
import os
import time

from joins.base_logger import logger
from joins.stats.evaluate_stats import evaluate_stats
from joins.stats.train_stats import train_stats
from joins.tools import convert_time_to_int

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train models', action='store_true')
    parser.add_argument(
        '--evaluate', help='evaluate models', action='store_true')
    parser.add_argument('--dataset', default='stats',
                        help='the target dataset')
    parser.add_argument('--data_folder',
                        default='data/stats/')
    parser.add_argument(
        '--model_folder', default='models', help="the folder to store model")
    parser.add_argument(
        '--model', default='models/stats_box_1024.pkl', help="the path to the pickled model, for evaluation")
    parser.add_argument(
        '--query', default='query.sql', help="the path to the query file, for evaluation")
    parser.add_argument(
        '--preprocess', help='convert date to int', action='store_true')
    parser.add_argument('--db_conn', type=str,
                        default="dbname=imdb user=postgres password=postgres host=127.0.0.1 port=5436",
                        help='postgres connection string')
    parser.add_argument(
        '--plot', help="display distribution plot during training", action='store_true')
    parser.add_argument(
        "--grid", help="grid size of model training", type=int, choices=range(5, 2**13), default=2**10)
    parser.add_argument(
        "--kernel", help="kernel function for density estimation", type=str, default='box')

    # parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)

    logger.info('Start running experiment for %s', args.dataset)

    if args.dataset == 'stats':
        if args.preprocess:
            logger.info("start pre-processing the data")
            convert_time_to_int(args.data_folder)

        elif args.train:
            logger.info("start training models")
            start_time = time.time()
            train_stats(args)
            end_time = time.time()
            logger.info(
                "Training completed: total training time is %.6f s.", end_time - start_time)

        elif args.evaluate:
            logger.info("start evaluating models")
            start_time = time.time()
            evaluate_stats(args)
            end_time = time.time()
            logger.info(
                "Evaluation completed: total evaluation time is %.6f s.", end_time - start_time)
