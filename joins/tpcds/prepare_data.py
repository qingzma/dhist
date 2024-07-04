from joins.schema_base import SchemaGraph, identify_key_values
from joins.tpcds.schema import gen_tpcds_light_schema


def process_tpcds_data(dataset, data_path, model_folder, kernel="gaussian"):
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_tpcds_light_schema(data_path)
    all_keys, equivalent_keys, table_keys = identify_key_values(schema)

    return schema, all_keys, equivalent_keys, table_keys
