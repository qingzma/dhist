import os
import pickle

from joins.base_logger import logger
from joins.schema_base import identify_conditions
from joins.stats.prepare_data import process_stats_data
from joins.stats.schema import get_stats_relevant_attributes
from joins.table_top_k import TableContainerTopK


def train_stats_topk(args):
    dataset = args.dataset
    data_path = args.data_folder
    model_folder = args.model_folder
    kernel = args.kernel
    grid = args.grid
    topk = args.topk

    model_container = dict()
    schema, all_keys, equivalent_keys, table_keys = process_stats_data(
        dataset, data_path, model_folder, kernel=kernel
    )
    join_keys, relevant_keys, counters = get_stats_relevant_attributes(schema)

    for t in schema.tables:
        print("table is ", t)
        table_path = os.path.join(data_path, t.table_name) + ".csv"
        logger.debug("training model for file %s", table_path)
        tableContainer = TableContainerTopK()
        tableContainer.fit(
            table_path,
            join_keys=join_keys,
            relevant_keys=relevant_keys,
            bin_info=schema.pk_bins,
            args=args,
        )
        model_container[t.table_name] = tableContainer

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    model_name = f"model_{dataset}_{grid}_{topk}"
    model_container["name"] = model_name
    model_container["schema"] = schema
    if args.cdf:
        model_name += "_cdf"
    model_path = os.path.join(model_folder, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(dict(model_container), f, pickle.HIGHEST_PROTOCOL)
    logger.info("models save at %s", model_path)
