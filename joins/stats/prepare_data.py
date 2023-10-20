from joins.stats.schema import gen_stats_light_schema
from joins.schema_base import SchemaGraph


def process_stats_data(dataset, data_path, model_folder, kernel='gaussian'):
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    all_keys, equivalent_keys, table_keys = identify_key_values(schema)

    return schema, all_keys, equivalent_keys, table_keys


def identify_key_values(schema: SchemaGraph):
    """
    identify all the key attributes from the schema of a DB, currently we assume all possible joins are known
    It is also easy to support unseen joins, which we left as a future work.
    :param schema: the schema of a DB
    :return: a dict of all keys, {table: [keys]};
             a dict of set, each indicating which keys on different tables are considered the same key.
             a dict of table keys.{table, set(join keys)}
    """
    all_keys = set()
    equivalent_keys = dict()
    table_keys = dict()
    for i, join in enumerate(schema.relationships):
        keys = join.identifier.split(" = ")
        all_keys.add(keys[0])
        all_keys.add(keys[1])
        seen = False
        for k in equivalent_keys:
            if keys[0] in equivalent_keys[k]:
                equivalent_keys[k].add(keys[1])
                seen = True
                break
            elif keys[1] in equivalent_keys[k]:
                equivalent_keys[k].add(keys[0])
                seen = True
                break
        if not seen:
            # set the keys[-1] as the identifier of this equivalent join key group for convenience.
            equivalent_keys[keys[-1]] = set(keys)

    assert len(all_keys) == sum(
        [len(equivalent_keys[k]) for k in equivalent_keys])

    for ks in all_keys:
        t, k = ks.split(".")[0], ks.split('.')[1]
        if t not in table_keys:
            table_keys[t] = set({k})
        else:
            table_keys[t].add(k)
    return all_keys, equivalent_keys, table_keys
