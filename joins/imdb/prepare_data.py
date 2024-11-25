from joins.schema_base import SchemaGraph, identify_key_values
from joins.imdb.schema import gen_job_light_imdb_schema


def process_imdb_data(dataset, data_path, model_folder, kernel='gaussian'):
    tables = ['movie_companies', 'movie_info_idx', 'title', 'movie_keyword', 'cast_info', 'movie_info']
    # if not data_path.endswith(".csv"):
    #     data_path += "/{}.csv".format()
    schema = gen_job_light_imdb_schema(data_path)
    all_keys, equivalent_keys, table_keys = identify_key_values(schema)

    return schema, all_keys, equivalent_keys, table_keys