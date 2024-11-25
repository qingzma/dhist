from joins.schema_base import SchemaGraph, Table


def gen_job_light_imdb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(
        Table(
            'title',
            primary_key=["id"],
            attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                        'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                        'series_years', 'md5sum'],
            irrelevant_attributes=['episode_of_id', 'title', 'imdb_index', 'phonetic_code', 'season_nr',
                                   'imdb_id', 'episode_nr', 'series_years', 'md5sum'],
            # no_compression=['kind_id'],
            csv_file_location=csv_path.format('title'),
            # table_size=3486660,
            table_size=2528312
            )
    )

    # movie_info_idx
    schema.add_table(
        Table(
            'movie_info_idx',
            primary_key=["id"],
            attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
            irrelevant_attributes=['info', 'note'],
            no_compression=['info_type_id'],
            csv_file_location=csv_path.format('movie_info_idx'),
            table_size=1380035)
    )

    # movie_info
    schema.add_table(
        Table('movie_info',
              primary_key=["id"],
              attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
              csv_file_location=csv_path.format('movie_info'),
              irrelevant_attributes=['info', 'note'],
              no_compression=['info_type_id'],
              table_size=14835720)
    )

    # cast_info
    schema.add_table(
        Table('cast_info',
              primary_key=["id"],
              attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                          'role_id'],
              csv_file_location=csv_path.format('cast_info'),
              irrelevant_attributes=['nr_order', 'note', 'person_id', 'person_role_id'],
              no_compression=['role_id'],
              table_size=36244344))

    # movie_keyword
    schema.add_table(
        Table('movie_keyword',
              primary_key=["id"],
              attributes=['id', 'movie_id', 'keyword_id'],
              csv_file_location=csv_path.format('movie_keyword'),
              no_compression=['keyword_id'],
              table_size=4523930)
    )

    # movie_companies
    schema.add_table(Table(
        'movie_companies',
        primary_key=["id"],
        attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
        csv_file_location=csv_path.format('movie_companies'),
        irrelevant_attributes=['note'],
        no_compression=['company_id', 'company_type_id'],
        table_size=2609129)
    )

    # relationships
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    # join key bin info
    # schema.add_pk_bins("title", "id", 1, 2528312)
    # schema.add_pk_bins("movie_info_idx", "movie_id", 2, 2525793)
    # schema.add_pk_bins("movie_info", "movie_id", 1, 2526430)
    # schema.add_pk_bins("cast_info", "movie_id", 1, 2525975)
    # schema.add_pk_bins("movie_keyword", "movie_id", 11, 2525971)
    # schema.add_pk_bins("movie_companies", "movie_id", 2, 2525745)
    schema.add_pk_bins("title", "id", 1, 2528312)
    schema.add_pk_bins("movie_info_idx", "movie_id", 1, 2528312)
    schema.add_pk_bins("movie_info", "movie_id", 1, 2528312)
    schema.add_pk_bins("cast_info", "movie_id", 1, 2528312)
    schema.add_pk_bins("movie_keyword", "movie_id", 1, 2528312)
    schema.add_pk_bins("movie_companies", "movie_id", 1, 2528312)

    # join paths
    schema.add_join_path(
        "SELECT COUNT(*) FROM  title t, movie_info_idx mi_idx, movie_info mi, cast_info ci, movie_keyword mk, movie_companies mc    WHERE  t.id = mi_idx.movie_id AND t.id=mi.movie_id AND t.id=ci.movie_id AND t.id=mk.movie_id  AND t.id=mc.movie_id"
    )

    return schema


def get_imdb_relevant_attributes(schema: SchemaGraph):
    join_keys, attrs, counters = dict(), dict(), dict()
    for k in schema.table_dictionary:
        atrributes = schema.table_dictionary[k].attributes
        primary_key = schema.table_dictionary[k].primary_key
        irrelevant_attributes = schema.table_dictionary[k].irrelevant_attributes
        relevant_attributes = [
            v
            for v in atrributes
            if (v not in primary_key and v not in irrelevant_attributes)
        ]
        attrs[k] = relevant_attributes
        counters[k] = schema.table_dictionary[k].table_size

        ids = [
            kk
            for kk in atrributes
            # if ("id" in kk and kk not in irrelevant_attributes)
            if (kk == "id" or kk == "movie_id")
        ]
        join_keys[k] = ids

    return join_keys, attrs, counters