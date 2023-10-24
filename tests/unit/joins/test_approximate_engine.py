# import unittest

# import numpy as np

# from joins.approximate_engine import ApproximateEngine


# class TestApproximateEngineMethod(unittest.TestCase):

#     def test_simple_query(self):
#         query = "SELECT COUNT(*) FROM votes as v, posts as p WHERE p.Id = v.PostId"
#         engine = ApproximateEngine()
#         tables_all, table_queries, join_cond, join_keys = parse_query_simple(
#             query)
#         key_conditions, non_key_conditions = identify_conditions(
#             table_queries, join_keys)
#         self.assertEqual(key_conditions, {})
#         self.assertEqual(non_key_conditions, {'posts': {'posts.Score': {
#                          '>=': -2}, 'posts.CommentCount': {'<=': 18}, 'posts.CreationDate': {'>=': 222608, '<=': 130899190}}})
