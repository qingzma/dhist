import argparse
import logging
import os
import time

from joins.tools import convert_time_to_int

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train models', action='store_true')
    parser.add_argument(
        '--evaluate', help='evaluate models', action='store_true')
    parser.add_argument('--dataset', default='stats',
                        help='the target dataset')
    parser.add_argument('--data_folder',
                        default='/home/quincy/Document/research/End-to-End-CardEst-Benchmark/datasets/stats_simplified_int/')
    parser.add_argument(
        '--preprocess', help='convert date to int', action='store_true')
    parser.add_argument('--db_conn', type=str,
                        default="dbname=imdb user=postgres password=postgres host=127.0.0.1 port=5436",
                        help='postgres connection string')

    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                "logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])

    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger(__name__)

    logger.info(f'Start running experiment for {args.dataset}')

    if args.dataset == 'stats':
        if args.preprocess:
            logger.info("start pre-processing the data")
            convert_time_to_int(args.data_folder)

        elif args.train:
            logger.info("start train models")

        elif args.evaluate:
            logger.info("start evaluating models")
