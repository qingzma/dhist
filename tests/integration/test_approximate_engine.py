import os
import pickle
import time
import unittest

import numpy as np

from joins.approximate_engine import ApproximateEngine
from joins.args import parse_args
from joins.base_logger import logger
from joins.stats.train_stats import train_stats
from joins.tools import q_error


class TestApproximateEngineMethod(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_name = "model_stats_gaussian_1000"
        self.use_pushed_down = True
    # train needed models

    # @classmethod
    # def setUpClass(cls):
    #     # ['biweight', 'box', 'cosine', 'epa', 'exponential', 'gaussian', 'tri', 'tricube', 'triweight']
    #     arguments = ["--train", "--grid", "200",
    #                  "--kernel", "gaussian", ]
    #     args = parse_args(arguments)
    #     train_stats(args)

    # remove trained models for test purposes
    # @classmethod
    # def tearDownClass(cls):
    #     for file in os.listdir("models"):
    #         print("files: " + file)
    #         if "100" in file:
    #             os.remove("models/"+file)

    def test_single_table_no_selection(self):
        query = "SELECT COUNT(*) FROM badges as b"
        with open("models/"+self.model_name+".pkl", 'rb') as f:
            model = pickle.load(f)
        engine = ApproximateEngine(model)
        t1 = time.time()
        res = engine.query_with_pushed_down(
            query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 79851
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2-t1)
        self.assertTrue(q_error(res, truth) < 1.01)

    def test_single_table_1_selection(self):
        query = "SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.AnswerCount<=4"
        with open("models/"+self.model_name+".pkl", 'rb') as f:
            model = pickle.load(f)
        engine = ApproximateEngine(model)
        t1 = time.time()
        res = engine.query_with_pushed_down(
            query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 42238
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2-t1)
        self.assertTrue(q_error(res, truth) < 3)

    def test_single_table_2_selections(self):
        query = "SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.AnswerCount<=4 AND p.CommentCount>=0 AND p.CommentCount<=17"
        with open("models/"+self.model_name+".pkl", 'rb') as f:
            model = pickle.load(f)
        engine = ApproximateEngine(model)
        t1 = time.time()
        res = engine.query_with_pushed_down(
            query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 42172
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2-t1)
        self.assertTrue(q_error(res, truth) < 5)

    def test_single_table_2_selections_1(self):
        query = "SELECT COUNT(*) FROM users as u WHERE u.DownVotes<=0 AND u.UpVotes>=0 AND u.UpVotes<=123"
        with open("models/"+self.model_name+".pkl", 'rb') as f:
            model = pickle.load(f)
        engine = ApproximateEngine(model)
        t1 = time.time()
        res = engine.query_with_pushed_down(
            query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 39532
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2-t1)
        self.assertTrue(q_error(res, truth) < 5)

    def test_single_table_3_selections(self):
        query = "SELECT COUNT(*) FROM users as u WHERE u.Reputation>=1 AND u.Reputation<=487 AND u.UpVotes<=27 AND u.CreationDate>='2010-10-22 22:40:35'::timestamp AND u.CreationDate<='2014-09-10 17:01:31'::timestamp"
        with open("models/"+self.model_name+".pkl", 'rb') as f:
            model = pickle.load(f)
        engine = ApproximateEngine(model)
        t1 = time.time()
        res = engine.query_with_pushed_down(
            query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 38103
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2-t1)
        self.assertTrue(q_error(res, truth) < 4)

    def test_single_table_more_selections(self):
        query = "SELECT COUNT(*) FROM users as u WHERE u.Reputation>=1 AND u.Views>=0 AND u.DownVotes>=0 AND u.UpVotes>=0 AND u.UpVotes<=15 AND u.CreationDate>='2010-09-03 11:45:16'::timestamp AND u.CreationDate<='2014-08-18 17:19:53'::timestamp"
        with open("models/"+self.model_name+".pkl", 'rb') as f:
            model = pickle.load(f)
        engine = ApproximateEngine(model)
        t1 = time.time()
        res = engine.query_with_pushed_down(
            query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 36820
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2-t1)
        self.assertTrue(q_error(res, truth) < 4)

    # def test_simple_query(self):
    #     query = "SELECT COUNT(*) FROM votes as v, posts as p WHERE p.Id = v.PostId"
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = ApproximateEngine(model)
    #     t1 = time.time()
    #     res = engine.query_with_pushed_down(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 328064
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 2)

    # def test_one_selection_query(self):
    #     query = "SELECT COUNT(*) FROM users as u, badges as b WHERE b.UserId= u.Id AND u.UpVotes>=0"
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = ApproximateEngine(model)
    #     t1 = time.time()
    #     res = engine.query_with_pushed_down(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 79851
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_multiple_table_same_join_column(self):
    #     query = "SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE c.UserId = u.Id AND b.UserId = u.Id"
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = ApproximateEngine(model)
    #     t1 = time.time()
    #     res = engine.query_with_pushed_down(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 15900001
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_push_down_query(self):
    #     query = "SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE c.UserId = u.Id AND b.UserId = u.Id AND b.Date<='2014-09-11 14:33:06'::timestamp AND c.Score>=0 AND c.Score<=10"
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = ApproximateEngine(model)
    #     t1 = time.time()
    #     res = engine.query_with_pushed_down(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 15852962
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)


if __name__ == '__main__':
    unittest.main()
