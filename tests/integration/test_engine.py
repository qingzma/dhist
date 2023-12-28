import os
import pickle
import time
import unittest

import numpy as np

from joins.engine import Engine
from joins.args import parse_args
from joins.base_logger import logger
from joins.stats.train_stats import train_stats
from joins.tools import q_error


class TestApproximateEngineMethod(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_name = "model_stats_gaussian_1000_cdf"
        self.use_pushed_down = True
        arguments = [
            "--train",
            "--grid",
            "1000",
            "--kernel",
            "gaussian",
            "--cdf",
        ]  # "--cdf"
        self.args = parse_args(arguments)

    # train needed models

    # @classmethod
    # def setUpClass(cls):
    #     # ['biweight', 'box', 'cosine', 'epa', 'exponential', 'gaussian', 'tri', 'tricube', 'triweight']
    #     arguments = [
    #         "--train",
    #         "--grid",
    #         "2000",
    #         "--kernel",
    #         "gaussian",
    #         "--cdf",
    #     ]  # "--cdf"
    #     args = parse_args(arguments)
    #     train_stats(args)

    # # remove trained models for test purposes
    # # @classmethod
    # # def tearDownClass(cls):
    # #     for file in os.listdir("models"):
    # #         print("files: " + file)
    # #         if "100" in file:
    # #             os.remove("models/"+file)

    # def test_single_table_no_selection(self):
    #     query = "SELECT COUNT(*) FROM badges as b"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 79851
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 1.01)

    # def test_single_table_1_selection(self):
    #     query = "SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.AnswerCount<=4"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 42238
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_single_table_2_selections(self):
    #     query = "SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.AnswerCount<=4 AND p.CommentCount>=0 AND p.CommentCount<=17"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 42172
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_single_table_2_selections_1(self):
    #     query = "SELECT COUNT(*) FROM users as u WHERE u.DownVotes<=0 AND u.UpVotes>=0 AND u.UpVotes<=123"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 39532
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_single_table_3_selections(self):
    #     query = "SELECT COUNT(*) FROM users as u WHERE u.Reputation>=1 AND u.Reputation<=487 AND u.UpVotes<=27 AND u.CreationDate>='2010-10-22 22:40:35'::timestamp AND u.CreationDate<='2014-09-10 17:01:31'::timestamp"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 38103
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 4)

    # def test_single_table_more_selections(self):
    #     query = "SELECT COUNT(*) FROM users as u WHERE u.Reputation>=1 AND u.Views>=0 AND u.DownVotes>=0 AND u.UpVotes>=0 AND u.UpVotes<=15 AND u.CreationDate>='2010-09-03 11:45:16'::timestamp AND u.CreationDate<='2014-08-18 17:19:53'::timestamp"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 36820
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 4)

    # def test_single_table_with_equality_condition_simple(self):
    #     query = "SELECT COUNT(*) FROM postHistory as ph WHERE  ph.CreationDate>='2010-09-14 11:59:07'::timestamp"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 297438
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 4)

    # def test_single_table_with_equality_condition(self):
    #     query = "SELECT COUNT(*) FROM postHistory as ph WHERE ph.PostHistoryTypeId=1 AND ph.CreationDate>='2010-09-14 11:59:07'::timestamp"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 42308
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 4)

    # def test_single_table_large_error(self):
    #     query = "SELECT COUNT(*) FROM votes as v WHERE v.BountyAmount>=0 AND v.CreationDate>='2010-07-20 00:00:00'::timestamp AND v.CreationDate<='2014-09-11 00:00:00'::timestamp"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 1740
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 4)

    # def test_single_table_small_error(self):
    #     query = "SELECT COUNT(*) FROM postHistory as ph WHERE ph.PostHistoryTypeId=5 AND ph.CreationDate>='2011-01-31 15:35:37'::timestamp"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 66005
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 4)

    # def test_single_table_small_error_strong_correlation(self):
    #     query = "SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.FavoriteCount>=0"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 13246
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 4)

    # def test_simple_join(self):
    #     query = "SELECT COUNT(*) FROM votes as v, posts as p WHERE p.Id = v.PostId"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 328064
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_simple_join1(self):
    #     query = "SELECT COUNT(*) FROM badges as b,  users as u WHERE  b.UserId = u.Id"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 79851
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_simple_join2(self):
    #     query = "SELECT COUNT(*) FROM comments as c, users as u WHERE c.UserId = u.Id "
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 171470
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_one_selection_query(self):
    #     query = "SELECT COUNT(*) FROM users as u, badges as b WHERE b.UserId= u.Id AND u.UpVotes>=0"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 79851
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_multiple_table_same_join_column(self):
    #     query = "SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE c.UserId = u.Id AND b.UserId = u.Id"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 15900001
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_push_down_query(self):
    #     query = "SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE c.UserId = u.Id AND b.UserId = u.Id AND b.Date<='2014-09-11 14:33:06'::timestamp AND c.Score>=0 AND c.Score<=10"
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 15852962
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_multi_way_2_join_key(self):
    #     # query = """SELECT COUNT(*) FROM postLinks as pl,  comments as c, posts as p,  users as u  WHERE c.UserId = u.Id  AND p.Id = pl.PostId  AND p.OwnerUserId = u.Id  AND p.CommentCount<=18  AND p.CreationDate>='2010-07-23 07:27:31'::timestamp  AND p.CreationDate<='2014-09-09 01:43:00'::timestamp"""
    #     query = """SELECT COUNT(*) FROM postLinks as pl,  posts as p,  users as u  WHERE p.Id = pl.PostId  AND p.OwnerUserId = u.Id  AND p.CommentCount<=18  AND p.CreationDate>='2010-07-23 07:27:31'::timestamp  AND p.CreationDate<='2014-09-09 01:43:00'::timestamp """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     # truth = 699302
    #     truth = 10826
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_2_join_key_1(self):
    #     query = """SELECT COUNT(*) FROM users as u, comments as c, posts as p WHERE p.OwnerUserId = u.Id AND p.Id = c.PostId AND u.UpVotes>=0 AND u.CreationDate>='2010-08-21 21:27:38'::timestamp AND c.CreationDate>='2010-07-21 11:05:37'::timestamp AND c.CreationDate<='2014-08-25 17:59:25'::timestamp"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 142137
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 10)

    # def test_multi_way_2_join_key_2(self):
    #     query = """SELECT COUNT(*) FROM users as u, comments as c, votes as v WHERE u.Id = c.UserId AND u.Id = v.UserId AND u.UpVotes>=0 AND u.UpVotes<=12 AND u.CreationDate>='2010-07-19 19:09:39'::timestamp AND c.Score=0 AND c.CreationDate<='2014-09-13 20:12:15'::timestamp AND v.BountyAmount<=50 AND v.CreationDate<='2014-09-12 00:00:00'::timestamp """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 2489
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_2_join_key_3(self):
    #     query = """SELECT COUNT(*)  FROM badges as b,  comments as c,  postLinks as pl,  posts as p,  users as u  WHERE u.Id = p.OwnerUserId  AND p.Id = pl.RelatedPostId  AND p.Id = c.PostId  AND u.Id = b.UserId  AND c.CreationDate<='2014-09-08 15:58:08'::timestamp AND p.ViewCount>=0"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 913441
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_2_join_key_4(self):
    #     query = """SELECT COUNT(*) FROM users as u, posts as p, postLinks as pl, badges as b WHERE p.Id = pl.RelatedPostId AND u.Id = p.OwnerUserId AND u.Id = b.UserId AND u.Views>=0 AND u.DownVotes>=0 AND u.CreationDate>='2010-08-04 16:59:53'::timestamp AND u.CreationDate<='2014-07-22 15:15:22'::timestamp AND p.CommentCount>=0 AND p.CommentCount<=13 AND b.Date<='2014-09-09 10:24:35'::timestamp"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 167260
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_2_join_key_hard(self):
    #     query = """SELECT COUNT(*) FROM users as u, comments as c, postHistory as ph, badges as b, votes as v WHERE c.UserId = u.Id AND b.UserId = u.Id AND ph.UserId = u.Id AND v.UserId = u.Id AND u.UpVotes=0 AND c.CreationDate>='2010-07-20 21:37:31'::timestamp AND ph.PostHistoryTypeId=12"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 224
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 40)

    # def test_multi_way_2_join_key_hard_reduce1(self):
    #     query = """SELECT COUNT(*) FROM  users as u,  comments as c,  votes as v , badges as b WHERE c.UserId = u.Id  AND v.UserId = u.Id  AND b.UserId = u.Id  AND u.UpVotes=0  AND c.CreationDate>='2010-07-20 21:37:31'::timestamp """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 11645
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    def test_multi_way_2_join_key_hard_reduce2(self):
        query = """SELECT COUNT(*) FROM  users as u,  comments as c,  votes as v  WHERE c.UserId = u.Id  AND v.UserId = u.Id  AND u.UpVotes=0  AND c.CreationDate>='2010-07-20 21:37:31'::timestamp   """
        with open("models/" + self.model_name + ".pkl", "rb") as f:
            model = pickle.load(f)
        engine = Engine(model, use_cdf=self.args.cdf)
        t1 = time.time()
        res = engine.query(
            query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 3672
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2 - t1)
        self.assertTrue(q_error(res, truth) < 5)

        # BUG JOIN KEY NULL VALUE noT FILTERED OUT!!
        # select count(*) from votes where userid is not NULL;
        # count
        # -------
        # 34789
        # (1 row)

        # select count(*) from votes where userid>=3 and userid<=55706;
        # count
        # -------
        # 34773
        # (1 row)

    # def test_multi_way_2_join_key_hard_reduce3(self):
    #     query = """SELECT COUNT(*) FROM  users as u,  comments as c WHERE c.UserId = u.Id  AND u.UpVotes=0  AND c.CreationDate>='2010-07-20 21:37:31'::timestamp  """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 20475
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_2_join_key_hard_reduce4(self):
    #     query = """SELECT COUNT(*) FROM  users as u WHERE u.UpVotes=0  """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 31529
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_1_join_key_hard1(self):
    #     query = """SELECT COUNT(*) FROM users as u, comments as c, badges as b, votes as v WHERE c.UserId = u.Id AND b.UserId = u.Id AND v.UserId = u.Id AND u.UpVotes=0 AND c.CreationDate>='2010-07-20 21:37:31'::timestamp"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 11645
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_1_join_key_hard2(self):
    #     query = """SELECT COUNT(*) FROM users as u, posts as p, postLinks as pl, badges as b WHERE p.Id = pl.RelatedPostId AND u.Id = p.OwnerUserId AND u.Id = b.UserId AND u.Views>=0 AND u.DownVotes>=0 AND u.CreationDate>='2010-08-04 16:59:53'::timestamp AND u.CreationDate<='2014-07-22 15:15:22'::timestamp AND p.CommentCount>=0 AND p.CommentCount<=13 AND b.Date<='2014-09-09 10:24:35'::timestamp"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 167260
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_single_table_test_123(self):
    #     query = """SELECT COUNT(*) FROM votes as v where v.BountyAmount<=50 AND v.CreationDate<='2014-09-12 00:00:00'::timestamp """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 1301
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_test(self):
    #     query = "SELECT COUNT(*) FROM badges as b where b.Date<='2014-09-11 14:33:06'::timestamp"
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 79633
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_test1(self):
    #     query = "SELECT COUNT(*) FROM comments as c where c.Score>=0 AND c.Score<=10"
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 174151
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_single_table_0_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p"""
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 91976
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_single_table_1_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p WHERE  p.PostTypeId=1"""
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 42921
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_single_table_2_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p WHERE  p.PostTypeId=1  AND p.Score<=35 """
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 42804
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_single_table_3_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p WHERE  p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1   """
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 17870
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_single_table_4_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p WHERE  p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17     """
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 17839
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_single_table_5_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p WHERE  p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17  AND p.FavoriteCount>=0  """
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 5209
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_setting_1(self):
    #     query = """SELECT COUNT(*) FROM  posts as p,   users as u  WHERE u.Id = p.OwnerUserId  AND p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17  AND p.FavoriteCount>=0  """
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 5121
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_setting_2(self):
    #     query = """SELECT COUNT(*) FROM  badges as b,  posts as p,  users as u  WHERE  u.Id = b.UserId  AND u.Id = p.OwnerUserId  AND b.Date>='2010-07-27 17:58:45'::timestamp  AND b.Date<='2014-09-06 17:33:22'::timestamp  AND p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17  AND p.FavoriteCount>=0  """
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 73766
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_setting_3(self):
    #     query = """SELECT COUNT(*) FROM  badges as b,  comments as c,  posts as p,  users as u  WHERE u.Id = c.UserId  AND u.Id = b.UserId  AND u.Id = p.OwnerUserId  AND b.Date>='2010-07-27 17:58:45'::timestamp  AND b.Date<='2014-09-06 17:33:22'::timestamp  AND p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17  AND p.FavoriteCount>=0   """
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 24666983
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_setting_4(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, posts as p, postHistory as ph, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = ph.UserId AND u.Id = p.OwnerUserId AND b.Date>='2010-07-27 17:58:45'::timestamp AND b.Date<='2014-09-06 17:33:22'::timestamp AND p.PostTypeId=1 AND p.Score<=35 AND p.AnswerCount=1 AND p.CommentCount<=17 AND p.FavoriteCount>=0"""
    #     with open("models/"+self.model_name+".pkl", 'rb') as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 64145760515
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2-t1)
    #     self.assertTrue(q_error(res, truth) < 5)


if __name__ == "__main__":
    unittest.main()
