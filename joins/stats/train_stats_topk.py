import os
import pickle

from joins.base_logger import logger
from joins.engine_topk import EngineTopK
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

    model_name = f"model_{dataset}_{grid}_{topk}"
    if args.cdf:
        model_name += "_cdf"
    model_path = os.path.join(model_folder, f"{model_name}.pkl")

    if os.path.exists(model_path):
        logger.warning("model[%s] already exists, skip training.", model_name)
    else:
        model_container = dict()
        schema, all_keys, equivalent_keys, table_keys = process_stats_data(
            dataset, data_path, model_folder, kernel=kernel
        )
        join_keys, relevant_keys, counters = get_stats_relevant_attributes(schema)

        for t in schema.tables:
            print("analyze table ", t.table_name)
            table_path = os.path.join(data_path, t.table_name) + ".csv"
            # logger.debug("training model for file %s", table_path)
            tableContainer = TableContainerTopK()
            tableContainer.fit(
                table_path,
                join_keys=join_keys,
                relevant_keys=relevant_keys,
                bin_info=schema.pk_bins,
                schema=schema,
                args=args,
            )
            model_container[t.table_name] = tableContainer

        model_container["name"] = model_name
        model_container["schema"] = schema

        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        with open(model_path, "wb") as f:
            pickle.dump(dict(model_container), f, pickle.HIGHEST_PROTOCOL)
        logger.info("models save at %s", model_path)

    if os.path.exists(model_path + ".dom"):
        logger.warning("model[%s] has dominating model, skip training.")
    else:
        with open("models/" + model_name + ".pkl", "rb") as f:
            model = pickle.load(f)

        tbl = list(model.keys())[0]
        if not model[tbl].jk_corrector:
            schema, all_keys, equivalent_keys, table_keys = process_stats_data(
                dataset, data_path, model_folder, kernel=kernel
            )
            join_keys, relevant_keys, counters = get_stats_relevant_attributes(schema)

            for query in schema.join_paths:
                engine = EngineTopK(model, use_cdf=args.cdf)
                top_container, join_path = engine.query(
                    query, n_dominating_counter=4000
                )

                logger.info("top_container %s", top_container)
                logger.info("join_path %s", join_path)

                # ["posts"]:  # : [i.table_name for i in schema.tables]
                for t in [i.table_name for i in schema.tables]:
                    table_path = os.path.join(data_path, t) + ".csv"
                    jks = [jk for jk in join_path if t in jk]
                    logger.info("table is %s", t)
                    logger.info("jks %s", jks)
                    assert len(jks) <= 1
                    if len(jks) == 1:
                        logger.info("jks is %s", jks)
                        jks = jks[0].replace(t, "").replace(".", "")
                        model[t].fit_join_key_corrector(
                            table_path,
                            join_keys=join_keys,
                            relevant_keys=relevant_keys,
                            bin_info=schema.pk_bins,
                            top_container=top_container,
                            join_path=join_path,
                            jks=jks,
                            args=args,
                        )
            with open(model_path, "wb") as f:
                pickle.dump(dict(model), f, pickle.HIGHEST_PROTOCOL)
            logger.info("models save at %s", model_path)
