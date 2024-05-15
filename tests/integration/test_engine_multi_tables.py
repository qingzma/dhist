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
    #         "1000",
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
    #     self.assertTrue(q_error(res, truth) < 10)

    # def test_multi_way_no_selection_2_u_c(self):
    #     query = """SELECT COUNT(*) FROM  comments as c,  users as u WHERE u.Id = c.UserId  """
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
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_no_selection_2_u_b(self):
    #     query = (
    #         """SELECT COUNT(*) FROM badges as b, users as u WHERE  u.Id = b.UserId  """
    #     )
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
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_no_selection_2_u_p(self):
    #     query = """SELECT COUNT(*) FROM  posts as p, users as u WHERE  u.Id = p.OwnerUserId """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 90584
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_no_selection_2_u_ph(self):
    #     query = """SELECT COUNT(*) FROM postHistory as ph, users as u WHERE  u.Id = ph.UserId """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 281859
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_no_selection_2_b_c(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c WHERE c.UserId = b.UserId """
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
    #     self.assertTrue(q_error(res, truth) < 15)

    # def test_multi_way_no_selection_2_b_p(self):
    #     query = """SELECT COUNT(*) FROM badges as b, posts as p WHERE  b.UserId= p.OwnerUserId """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 3728360
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 10)

    # def test_multi_way_no_selection_2_b_ph(self):
    #     query = """SELECT COUNT(*) FROM badges as b,  postHistory as ph WHERE b.UserId = ph.UserId  """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 16322646
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 10)

    # def test_multi_way_no_selection_2_c_p(self):
    #     query = """SELECT COUNT(*) FROM  comments as c, posts as p WHERE c.UserId  = p.OwnerUserId """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 56398574
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 10)

    # def test_multi_way_no_selection_2_c_ph(self):
    #     query = """SELECT COUNT(*) FROM  comments as c,  postHistory as ph WHERE c.UserId  = ph.UserId  """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 263105194
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 15)

    # def test_multi_way_no_selection_2_p_ph(self):
    #     query = """SELECT COUNT(*) FROM  posts as p, postHistory as ph WHERE  ph.UserId = p.OwnerUserId """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 54807156
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_no_selection_3(self):
    #     query = """SELECT COUNT(*)  FROM badges as b,  posts as p,  users as u  WHERE u.Id = p.OwnerUserId   AND u.Id = b.UserId"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 3728360
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_way_no_selection_4(self):
    #     query = """SELECT COUNT(*)  FROM badges as b,  comments as c,  posts as p,  users as u  WHERE u.Id = p.OwnerUserId    AND u.Id = c.UserId  AND u.Id = b.UserId"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 15131840763
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 100)

    # def test_2_table_4_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p, users as u WHERE    u.Id = p.OwnerUserId  AND  p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount<=5  AND p.CommentCount<=17     """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 41718
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_2_table_4_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p, users as u WHERE    u.Id = p.OwnerUserId  AND  p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount<=5  AND p.CommentCount<=17    AND u.UpVotes<=50 """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 36120
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    # def test_2_table_4_selection(self):
    #     query = """SELECT COUNT(*) FROM  posts as p, users as u WHERE    u.Id = p.OwnerUserId  AND  p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount<=5  AND p.CommentCount<=17    AND u.UpVotes<=50 AND u.DownVotes<=50"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 36113
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 3)

    def test_real1(self):
        query = """SELECT COUNT(*) FROM comments as c, posts as p, postHistory as ph, badges as b, users as u WHERE u.Id = ph.UserId AND u.Id = b.UserId AND u.Id = p.OwnerUserId AND u.Id = c.UserId AND c.Score=0 AND p.Score>=-2 AND p.CommentCount>=0 AND p.CommentCount<=12 AND p.FavoriteCount>=0 AND p.FavoriteCount<=6 AND ph.CreationDate<='2014-08-18 08:54:12'::timestamp AND u.Views=0 AND u.DownVotes>=0 AND u.DownVotes<=60"""
        with open("models/" + self.model_name + ".pkl", "rb") as f:
            model = pickle.load(f)
        engine = Engine(model, use_cdf=self.args.cdf)
        t1 = time.time()
        res = engine.query(query) if self.use_pushed_down else engine.query(query)
        t2 = time.time()
        truth = 16698
        logger.info("result %.6E", res)
        logger.info("truth %.6E", truth)
        logger.info("time cost is %.5f s.", t2 - t1)
        self.assertTrue(q_error(res, truth) < 10)

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
    #     self.assertTrue(q_error(res, truth) < 20)

    # # def test_multi_way_2_join_key_hard(self):
    # #     query = """SELECT COUNT(*) FROM users as u, comments as c, postHistory as ph, badges as b, votes as v WHERE c.UserId = u.Id AND b.UserId = u.Id AND ph.UserId = u.Id AND v.UserId = u.Id AND u.UpVotes>=50 AND c.CreationDate>='2010-07-20 21:37:31'::timestamp AND ph.PostHistoryTypeId>=5"""
    # #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    # #         model = pickle.load(f)
    # #     engine = Engine(model, use_cdf=self.args.cdf)
    # #     t1 = time.time()
    # #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    # #     t2 = time.time()
    # #     truth = 32734
    # #     logger.info("result %.6E", res)
    # #     logger.info("truth %.6E", truth)
    # #     logger.info("time cost is %.5f s.", t2 - t1)
    # #     self.assertTrue(q_error(res, truth) < 40)

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

    # def test_multi_way_2_join_key_hard_reduce2(self):
    #     query = """SELECT COUNT(*) FROM  users as u,  comments as c,  votes as v  WHERE c.UserId = u.Id  AND v.UserId = u.Id  AND u.UpVotes=0  AND c.CreationDate>='2010-07-20 21:37:31'::timestamp   """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 3672
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

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
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
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
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
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

    # def test_multi_table_setting_1(self):
    #     query = """SELECT COUNT(*) FROM  posts as p,   users as u  WHERE u.Id = p.OwnerUserId  AND p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17  AND p.FavoriteCount>=0  """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 5121
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_setting_2(self):
    #     query = """SELECT COUNT(*) FROM  badges as b,  posts as p,  users as u  WHERE  u.Id = b.UserId  AND u.Id = p.OwnerUserId  AND b.Date>='2010-07-27 17:58:45'::timestamp  AND b.Date<='2014-09-06 17:33:22'::timestamp  AND p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17  AND p.FavoriteCount>=0  """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 73766
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 20)

    # def test_multi_table_setting_3(self):
    #     query = """SELECT COUNT(*) FROM  badges as b,  comments as c,  posts as p,  users as u  WHERE u.Id = c.UserId  AND u.Id = b.UserId  AND u.Id = p.OwnerUserId  AND b.Date>='2010-07-27 17:58:45'::timestamp  AND b.Date<='2014-09-06 17:33:22'::timestamp  AND p.PostTypeId=1  AND p.Score<=35  AND p.AnswerCount=1  AND p.CommentCount<=17  AND p.FavoriteCount>=0   """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 24666983
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 20)

    # def test_multi_table_setting_4(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, posts as p, postHistory as ph, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = ph.UserId AND u.Id = p.OwnerUserId AND b.Date>='2010-07-27 17:58:45'::timestamp AND b.Date<='2014-09-06 17:33:22'::timestamp AND p.PostTypeId=1 AND p.Score<=35 AND p.AnswerCount=1 AND p.CommentCount<=17 AND p.FavoriteCount>=0"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 64145760515
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_2(self):
    #     query = (
    #         """SELECT COUNT(*) FROM badges as b, users as u WHERE  u.Id = b.UserId """
    #     )
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
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_3(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId  """
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
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_4(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c,posts as p,  users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = p.OwnerUserId """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 15131840763
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_5(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, posts as p, postHistory as ph, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = ph.UserId AND u.Id = p.OwnerUserId"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 15131840763
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_2_with_single_condition_less_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, users as u WHERE  u.Id = b.UserId AND b.Date>='2013-07-20 19:02:22'::timestamp and u.UpVotes<=1000"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 35227  # 1379  #
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_2_with_single_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, users as u WHERE  u.Id = b.UserId AND b.Date>='2013-07-20 19:02:22'::timestamp and u.UpVotes>=5"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 13391
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_3_with_single_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND b.Date>='2013-07-20 19:02:22'::timestamp and u.UpVotes>=5 and c.Score=0 """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 4260125  # 67211  #
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_3_with_single_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND b.Date>='2013-07-20 19:02:22'::timestamp and u.UpVotes>=10 and c.Score=0 """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 4229950  # 67211  #
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # # # this is a bug !!!!!!!!!!!!!!!!!!!!!!!!!!!!! only upvotes as 5 seem good enough
    # def test_multi_table_same_join_column_3_with_single_condition_less_than_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND b.Date>='2013-07-20 19:02:22'::timestamp and u.UpVotes<1 and c.Score=0 """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 67211  # 67211  # 4260125  # 67211  #
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 10)

    # def test_multi_table_same_join_column_3_with_multi_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND b.Date>='2013-07-20 19:02:22'::timestamp  and u.UpVotes>=2 and c.Score=0 """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 4285046
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 6)

    # def test_multi_table_same_join_column_4_with_single_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c,posts as p,  users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = p.OwnerUserId AND b.Date>='2013-07-20 19:02:22'::timestamp and u.UpVotes>=5 and c.Score=10  and p.CreationDate>='2013-07-23 07:27:31'::timestamp """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 981758  # 2102603883
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_5_with_single_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, posts as p, postHistory as ph, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = ph.UserId AND u.Id = p.OwnerUserId  AND b.Date>='2013-07-20 19:02:22'::timestamp and u.UpVotes>=5 and c.Score=10  and p.CreationDate>='2013-07-23 07:27:31'::timestamp and ph.CreationDate>='2013-07-23 11:59:07'::timestamp"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 3244103433
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 20)

    # def test_multi_table_same_join_column_2_with_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, users as u WHERE u.Id = b.UserId AND b.Date>='2010-07-27 17:58:45'::timestamp AND b.Date<='2014-09-06 17:33:22'::timestamp """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 78392
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_3_with_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, posts as p, users as u WHERE u.Id = b.UserId AND u.Id = p.OwnerUserId AND b.Date>='2010-07-27 17:58:45'::timestamp AND b.Date<='2014-09-06 17:33:22'::timestamp AND p.PostTypeId=1  """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 445767
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 10)

    # def test_multi_table_same_join_column_4_with_condition(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, posts as p, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = p.OwnerUserId AND b.Date>='2010-07-27 17:58:45'::timestamp AND b.Date<='2014-09-06 17:33:22'::timestamp AND p.PostTypeId=1 AND p.Score<=35 AND p.AnswerCount=1 AND p.CommentCount<=17 AND p.FavoriteCount>=0 AND c.Score>5 """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 154993
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_4_with_conditions(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, posts as p, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = p.OwnerUserId AND b.Date>='2010-07-27 17:58:45'::timestamp AND b.Date<='2014-09-06 17:33:22'::timestamp AND p.PostTypeId=1 AND p.Score<=35 AND p.AnswerCount=1 AND p.CommentCount<=17 AND p.FavoriteCount>=0 AND c.Score>5 """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 154993
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_key_large_errors(self):
    #     query = """SELECT COUNT(*) FROM users as u, comments as c, postHistory as ph, votes as v WHERE c.UserId = u.Id AND v.UserId = u.Id AND ph.UserId = u.Id AND u.Reputation>=1 AND u.Views>=0 AND u.Views<=110 AND u.UpVotes=0 AND u.CreationDate>='2010-07-28 19:29:11'::timestamp AND u.CreationDate<='2014-08-14 05:29:30'::timestamp AND v.BountyAmount>=0 AND v.CreationDate>='2010-07-26 00:00:00'::timestamp AND v.CreationDate<='2014-09-08 00:00:00'::timestamp"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 6299
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_key_small_errors(self):
    #     query = """SELECT COUNT(*) FROM votes as v, comments as c, postHistory as ph, users as u WHERE c.UserId = u.Id AND v.UserId = u.Id AND ph.UserId = u.Id AND v.CreationDate>='2010-07-26 00:00:00'::timestamp AND v.CreationDate<='2014-09-12 00:00:00'::timestamp AND c.CreationDate>='2010-08-12 20:33:46'::timestamp AND c.CreationDate<='2014-09-13 19:26:55'::timestamp AND ph.CreationDate>='2011-04-11 14:46:09'::timestamp AND ph.CreationDate<='2014-08-17 16:37:23'::timestamp"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(
    #         query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 52988984588
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_multi_table_same_join_column_large_error(self):
    #     query = """SELECT COUNT(*) FROM posts as p WHERE p.ViewCount>=0 AND p.AnswerCount<=5 AND p.CommentCount<=12 AND p.FavoriteCount>=0"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 12741
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # -------------------------------------------------------------------------
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
    #     self.assertTrue(q_error(res, truth) < 10)

    # --------------------------------------------------------------
    # TODO: NOT TESTABLE
    # --------------------------------------------------------------
    # def test_multi_way_no_selection_5(self):
    #     query = """SELECT COUNT(*) FROM badges as b, comments as c, posts as p, postHistory as ph, users as u WHERE u.Id = c.UserId AND u.Id = b.UserId AND u.Id = ph.UserId AND u.Id = p.OwnerUserId """
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 913994
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

    # def test_dual_join_key(self):
    #     query = """SELECT COUNT(*)  FROM badges as b,  comments as c,  postLinks as pl,  posts as p,  users as u  WHERE u.Id = p.OwnerUserId  AND p.Id = pl.RelatedPostId  AND p.Id = c.PostId  AND u.Id = b.UserId"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
    #     t2 = time.time()
    #     truth = 913994
    #     logger.info("result %.6E", res)
    #     logger.info("truth %.6E", truth)
    #     logger.info("time cost is %.5f s.", t2 - t1)
    #     self.assertTrue(q_error(res, truth) < 5)

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

    # def test_multi_way_2_join_key_3(self):
    #     query = """SELECT COUNT(*)  FROM badges as b,  comments as c,  postLinks as pl,  posts as p,  users as u  WHERE u.Id = p.OwnerUserId  AND p.Id = pl.RelatedPostId  AND p.Id = c.PostId  AND u.Id = b.UserId  AND c.CreationDate<='2014-09-08 15:58:08'::timestamp AND p.ViewCount>=0"""
    #     with open("models/" + self.model_name + ".pkl", "rb") as f:
    #         model = pickle.load(f)
    #     engine = Engine(model, use_cdf=self.args.cdf)
    #     t1 = time.time()
    #     res = engine.query(query) if self.use_pushed_down else engine.query(query)
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


if __name__ == "__main__":
    unittest.main()
