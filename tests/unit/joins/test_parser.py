import unittest

import numpy as np

from joins.parser import parse_query_simple, parse_single_table_query
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

    def test_parse_single_table_query(self):
        query = "SELECT COUNT(*) FROM votes as v"
        tbl_name, cols = parse_single_table_query(query)
        self.assertEqual(tbl_name, "votes")
        self.assertFalse(cols)

        query = "SELECT COUNT(*) FROM comments as c WHERE c.Score=0"
        tbl_name, cols = parse_single_table_query(query)
        self.assertEqual(tbl_name, "comments")
        self.assertEqual(cols, ["Score"])

        query = "SELECT COUNT(*) FROM badges as b WHERE b.Date<='2014-09-11 14:33:06'::timestamp"
        tbl_name, cols = parse_single_table_query(query)
        self.assertEqual(tbl_name, "badges")
        self.assertEqual(cols, ["Date"])

        query = "SELECT COUNT(*) FROM postHistory as ph WHERE ph.PostHistoryTypeId=1 AND ph.CreationDate>='2010-09-14 11:59:07'::timestamp"
        tbl_name, cols = parse_single_table_query(query)
        self.assertEqual(tbl_name, "postHistory")
        cols = ["PostHistoryTypeId", "CreationDate"]
        cols.sort()
        self.assertEqual(cols, cols)

        query = "SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.AnswerCount<=4 AND p.CommentCount>=0 AND p.CommentCount<=17"
        tbl_name, cols = parse_single_table_query(query)
        self.assertEqual(tbl_name, "posts")
        cols = ["CommentCount", "AnswerCount"]
        cols.sort()
        self.assertEqual(cols, cols)


if __name__ == '__main__':
    unittest.main()
