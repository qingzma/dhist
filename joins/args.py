import argparse


def parse_args(args_):
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
        "--grid", help="grid size of model training", type=int, choices=range(5, 2**20), default=2**10)
    parser.add_argument(
        "--kernel", help="kernel function for density estimation", type=str, default='box')

    # parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    args = parser.parse_args(args_)
    return args
