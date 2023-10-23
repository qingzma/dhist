import unittest

import numpy as np

from joins.parser import parse_query_simple
from joins.schema_base import identify_conditions


class TestParserMethod(unittest.TestCase):

    def test_parse_query_simple(self):
        query = "SELECT COUNT(*) FROM votes as v, posts as p WHERE p.Id = v.PostId AND p.Score>=-2 AND p.CommentCount<=18 AND p.CreationDate>='2010-07-21 13:50:08'::timestamp AND p.CreationDate<='2014-09-11 00:53:10'::timestamp"
        tables_all, table_queries, join_cond, join_keys = parse_query_simple(
            query)
        key_conditions, non_key_conditions = identify_conditions(
            table_queries, join_keys)
        self.assertEqual(key_conditions, {})
        self.assertEqual(non_key_conditions, {'posts': {'posts.Score': {
                         '>=': -2}, 'posts.CommentCount': {'<=': 18}, 'posts.CreationDate': {'>=': 222608, '<=': 130899190}}})
