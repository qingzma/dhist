import os
import pickle

from joins.base_logger import logger
from joins.stats.prepare_data import process_stats_data
from joins.table import TableContainer


def train_stats(args):
    dataset = args.dataset
    data_path = args.data_folder
    model_folder = args.model_folder
    kernel = args.kernel
    grid = args.grid
    # table = TableContainer()
    # table.fit('data/pm25_100.csv', join_keys=['PRES'])
    model_container = dict()
    schema, all_keys, equivalent_keys, table_keys = process_stats_data(
        dataset, data_path, model_folder, kernel=kernel)

    # print(data_path)
    for t in schema.tables:
        table_path = os.path.join(data_path, t.table_name) + '.csv'
        logger.debug("training model for file %s", table_path)
        # df = pd.read_csv(table_path)
        # print(df)
        tableContainer = TableContainer()
        tableContainer.fit(
            table_path, join_keys=table_keys[t.table_name], args=args)
        model_container[t.table_name] = tableContainer

    # print(schema)
    # print(all_keys)
    # print(equivalent_keys)
    # print(table_keys)

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_name = f"model_{dataset}_{kernel}_{grid}"
    model_container['name'] = model_name
    model_container['schema'] = schema
    model_path = os.path.join(
        model_folder, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_container, f, pickle.HIGHEST_PROTOCOL)
    logger.info("models save at %s", model_path)
