import os
import pickle
import unittest

import numpy as np

from joins.approximate_engine import ApproximateEngine
from joins.args import parse_args
from joins.stats.train_stats import train_stats


class TestApproximateEngineMethod(unittest.TestCase):
    # train needed models
    @classmethod
    def setUpClass(cls):
        arguments = ["--train", "--grid", "102"]
        args = parse_args(arguments)
        train_stats(args)

    # remove trained models for test purposes
    @classmethod
    def tearDownClass(cls):
        for file in os.listdir("models"):
            print("files: " + file)
            if "102" in file:
                os.remove("models/"+file)

    def test_simple_query(self):
        query = "SELECT COUNT(*) FROM votes as v, posts as p WHERE p.Id = v.PostId"
        with open("models/model_stats_box_102.pkl", 'rb') as f:
            model = pickle.load(f)
        engine = ApproximateEngine(model)
        engine.query(query)


if __name__ == '__main__':
    unittest.main()
