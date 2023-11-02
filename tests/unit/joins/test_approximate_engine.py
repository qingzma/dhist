import os
import pickle
import time
import unittest

import numpy as np

from joins.approximate_engine import (array_multiply_grid,
                                      combine_selectivity_grid_with_two_arrays,
                                      generate_push_down_conditions,
                                      grid_multiply_array, parse_query_simple)
from joins.args import parse_args
from joins.base_logger import logger
from joins.stats.train_stats import train_stats
from joins.tools import q_error


class TestApproximateEngineMethodUniteTest(unittest.TestCase):
    def test_single_table_query(self):
        query_str = "SELECT COUNT(*) FROM badges as b"
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys)
        # logger.info("condition is %s", conditions)
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions["badges"][0].join_keys, ["badges.Id"])
        self.assertIsNone(conditions['badges']
                          [0].non_key)
        self.assertIsNone(conditions['badges']
                          [0].non_key_condition)
        self.assertDictEqual(conditions['badges']
                             [0].to_join, {})

    def test_single_table_query1(self):
        query_str = "SELECT COUNT(*) FROM badges as b where b.col <= 20"
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys)
        logger.info("condition is %s", conditions)
        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions["badges"][0].join_keys, ["badges.Id"])
        self.assertEqual(conditions['badges']
                         [0].non_key, "badges.col")
        self.assertEqual(conditions['badges']
                         [0].non_key_condition, [-np.Infinity, 20])
        self.assertDictEqual(conditions['badges']
                             [0].to_join, {})

    def test_single_table_query2(self):
        query_str = "SELECT COUNT(*) FROM badges as b where b.col <= 20 AND b.col2 = 1"
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys)
        # logger.info("condition is %s", conditions)
        self.assertEqual(len(conditions["badges"]), 2)
        self.assertEqual(conditions["badges"][0].join_keys, ["badges.Id"])
        self.assertEqual(conditions['badges']
                         [0].non_key, "badges.col")
        self.assertEqual(conditions['badges']
                         [0].non_key_condition, [-np.Infinity, 20])
        self.assertEqual(conditions['badges']
                         [1].non_key_condition, [0.5, 1.5])
        self.assertDictEqual(conditions['badges']
                             [0].to_join, {})

    def test_multiple_condition_query(self):
        query_str = "SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE c.UserId = u.Id AND b.UserId = u.Id AND b.Date<='2014-09-11 14:33:06'::timestamp AND c.Score>=0 AND c.Score<=10"
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys)
        self.assertEqual(len(conditions), 3)
        self.assertEqual(len(conditions['comments']), 1)
        self.assertEqual(conditions['comments']
                         [0].join_keys, ["comments.UserId"])
        # self.assertEqual(conditions['comments']
        #                  [0].join_keys, ["comments.UserId"])
        self.assertEqual(conditions['comments']
                         [0].to_join, {'users': ['Id']})
        self.assertEqual(conditions['comments']
                         [0].non_key, "comments.Score")

    def test_multiple_condition_query1(self):
        query_str = "SELECT COUNT(*) FROM badges as b, comments as c, users as u " + \
            "WHERE c.UserId = u.Id AND b.UserId = u.Id AND b.Date<='2014-09-11 14:33:06'::timestamp " +\
            "AND c.Score>=0 AND c.Score<=10 AND c.hh<201"
        tables_all, table_query, join_cond, join_keys = parse_query_simple(
            query_str)
        conditions = generate_push_down_conditions(
            tables_all, table_query, join_cond, join_keys)
        # print("conditions", conditions)
        self.assertEqual(len(conditions), 3)
        self.assertEqual(len(conditions['comments']), 2)
        self.assertEqual(conditions['comments']
                         [0].join_keys, ["comments.UserId"])
        self.assertEqual(conditions['comments']
                         [0].join_keys, ["comments.UserId"])
        self.assertEqual(conditions['comments']
                         [0].to_join, {'users': ['Id']})
        self.assertEqual(conditions['comments']
                         [0].non_key, "comments.Score")
        self.assertEqual(conditions['comments']
                         [0].non_key_condition, [0, 10])
        self.assertEqual(conditions['comments']
                         [1].non_key_condition, [-np.Infinity, 201])

    def test_combine_selectivity_grid_with_two_arrays(self):
        b = np.array([
            [1, 2],
            [3, 4],
            [5, 6],])
        a = np.array([1, 2])
        c = np.array([1, 2, 3])
        res = combine_selectivity_grid_with_two_arrays(a, b, c)
        minused = res - np.array([[1,  4],
                                  [6, 16],
                                  [15, 36]])
        self.assertTrue(minused.all() == 0)

        res = array_multiply_grid(a, b)
        minused = res - np.array([9,  24])
        self.assertTrue(minused.all() == 0)

        res = grid_multiply_array(b, c)
        minused = res - np.array([22, 28])
        self.assertTrue(minused.all() == 0)


if __name__ == '__main__':
    unittest.main()
    # TestApproximateEngineMethodUniteTest().test_single_table_query()
