from joins.schema_base import SchemaGraph, Table


def gen_stats_light_schema(hdf_path):
    """
    Generate the stats schema with a small subset of data.
    """

    schema = SchemaGraph()

    # tables

    # badges
    schema.add_table(
        Table(
            "badges",
            primary_key=["Id"],
            attributes=["Id", "UserId", "Date"],
            irrelevant_attributes=[],  # ['Id'],
            no_compression=[],
            csv_file_location=hdf_path.format("badges"),
            table_size=79851,
        )
    )

    # votes
    schema.add_table(
        Table(
            "votes",
            primary_key=["Id"],
            attributes=[
                "Id",
                "PostId",
                "VoteTypeId",
                "CreationDate",
                "UserId",
                "BountyAmount",
            ],
            csv_file_location=hdf_path.format("votes"),
            irrelevant_attributes=[],  # ['Id'],
            no_compression=["VoteTypeId"],
            table_size=328064,
        )
    )

    # postHistory
    schema.add_table(
        Table(
            "postHistory",
            primary_key=["Id"],
            attributes=["Id", "PostHistoryTypeId", "PostId", "CreationDate", "UserId"],
            csv_file_location=hdf_path.format("postHistory"),
            irrelevant_attributes=[],  # ['Id'],
            no_compression=["PostHistoryTypeId"],
            table_size=303187,
        )
    )

    # posts
    schema.add_table(
        Table(
            "posts",
            primary_key=["Id"],
            attributes=[
                "Id",
                "PostTypeId",
                "CreationDate",
                "Score",
                "ViewCount",
                "OwnerUserId",
                "AnswerCount",
                "CommentCount",
                "FavoriteCount",
                "LastEditorUserId",
            ],
            csv_file_location=hdf_path.format("posts"),
            irrelevant_attributes=["LastEditorUserId"],
            no_compression=["PostTypeId"],
            table_size=91976,
        )
    )

    # users
    schema.add_table(
        Table(
            "users",
            primary_key=["Id"],
            attributes=[
                "Id",
                "Reputation",
                "CreationDate",
                "Views",
                "UpVotes",
                "DownVotes",
            ],
            csv_file_location=hdf_path.format("users"),
            no_compression=[],
            table_size=40325,
        )
    )

    # comments
    schema.add_table(
        Table(
            "comments",
            primary_key=["Id"],
            attributes=["Id", "PostId", "Score", "CreationDate", "UserId"],
            csv_file_location=hdf_path.format("comments"),
            irrelevant_attributes=[],  # ["Id"],
            no_compression=[],
            table_size=174305,
        )
    )

    # postLinks
    schema.add_table(
        Table(
            "postLinks",
            primary_key=["Id"],
            attributes=["Id", "CreationDate", "PostId", "RelatedPostId", "LinkTypeId"],
            csv_file_location=hdf_path.format("postLinks"),
            irrelevant_attributes=[],  # ["Id"],
            no_compression=[],
            table_size=11102,
        )
    )

    # tags
    schema.add_table(
        Table(
            "tags",
            attributes=["Id", "Count", "ExcerptPostId"],
            csv_file_location=hdf_path.format("tags"),
            irrelevant_attributes=[],  # ["Id"],
            no_compression=[],
            table_size=1032,
        )
    )

    # relationships
    schema.add_relationship("comments", "PostId", "posts", "Id")
    schema.add_relationship("comments", "UserId", "users", "Id")

    schema.add_relationship("badges", "UserId", "users", "Id")

    schema.add_relationship("tags", "ExcerptPostId", "posts", "Id")

    schema.add_relationship("postLinks", "PostId", "posts", "Id")
    schema.add_relationship("postLinks", "RelatedPostId", "posts", "Id")

    schema.add_relationship("postHistory", "PostId", "posts", "Id")
    schema.add_relationship("postHistory", "UserId", "users", "Id")
    schema.add_relationship("votes", "PostId", "posts", "Id")
    schema.add_relationship("votes", "UserId", "users", "Id")

    schema.add_relationship("posts", "OwnerUserId", "users", "Id")

    # join key bin info
    schema.add_pk_bins("users", "Id", -1, 55747)
    schema.add_pk_bins("badges", "UserId", -1, 55747)
    schema.add_pk_bins("postHistory", "UserId", -1, 55747)
    schema.add_pk_bins("votes", "UserId", -1, 55747)
    schema.add_pk_bins("comments", "UserId", -1, 55747)
    schema.add_pk_bins("posts", "OwnerUserId", -1, 55747)

    schema.add_pk_bins("posts", "Id", 1, 115378)
    schema.add_pk_bins("postLinks", "PostId", 1, 115378)
    schema.add_pk_bins("comments", "PostId", 1, 115378)
    schema.add_pk_bins("postHistory", "PostId", 1, 115378)
    schema.add_pk_bins("votes", "PostId", 1, 115378)
    schema.add_pk_bins("postLinks", "RelatedPostId", 1, 115378)
    schema.add_pk_bins("tags", "ExcerptPostId", 1, 115378)

    schema.add_join_path(
        "SELECT COUNT(*) FROM  users as u, badges as b, comments as c, votes as v, posts as p, postHistory as ph    WHERE  u.Id = b.UserId AND u.Id=c.UserId AND u.Id=v.UserId AND u.Id=p.OwnerUserId  AND u.Id=ph.UserId"
    )
    schema.add_join_path(
        "SELECT COUNT(*) FROM  posts as p, postLinks as pl, comments as c, postHistory as ph, votes as v      WHERE  p.Id = pl.PostId AND p.Id = c.PostId AND p.Id = ph.PostId AND p.Id = v.PostId"
    )
    schema.add_join_path(
        "SELECT COUNT(*) FROM  posts as p, postLinks as pl, comments as c, postHistory as ph, votes as v      WHERE  p.Id = pl.RelatedPostId AND p.Id = c.PostId AND p.Id = ph.PostId AND p.Id = v.PostId"
    )
    # schema.add_join_path(["posts.Id", "postLinks.PostId", "comments.PostId", "postHistory.PostId",
    #                       "votes.PostId", "tags.ExcerptPostId"])

    return schema


def get_stats_relevant_attributes(schema: SchemaGraph):
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
            if ("Id" in kk and kk not in irrelevant_attributes and kk != "LinkTypeId")
        ]
        join_keys[k] = ids

    return join_keys, attrs, counters


# def get_stats_relevant_attributes():
#     return dict({'badges': ['UserId', 'Date'],
#                  'votes': ['PostId', 'CreationDate', 'UserId', 'BountyAmount'],
#                  'postHistory': ['PostId', 'CreationDate', 'UserId'],
#                  'posts': ['PostTypeId', 'CreationDate',
#                            'Score', 'ViewCount', 'OwnerUserId',
#                            'AnswerCount', 'CommentCount', 'FavoriteCount', ],
#                  'users': ['Reputation', 'CreationDate',
#                            'Views', 'UpVotes', 'DownVotes'],
#                  'comments': ['PostId', 'Score',
#                               'CreationDate', 'UserId'],
#                  'postLinks': ['CreationDate',
#                                'PostId', 'RelatedPostId', 'LinkTypeId'],
#                  'tags': ['Id', 'Count', 'ExcerptPostId']
#                  })
