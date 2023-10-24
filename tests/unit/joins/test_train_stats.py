import os
import unittest

from joins.args import parse_args
from joins.stats.train_stats import train_stats


class TestTrainStats(unittest.TestCase):

    def test_train_stats(self):
        arguments = ["--train", "--grid", "102"]
        args = parse_args(arguments)
        train_stats(args)

    # remove trained models for test purposes
    @classmethod
    def tearDownClass(self):
        for file in os.listdir("models"):
            # print("remove file: models/" + file)
            if "102" in file:
                os.remove("models/"+file)


if __name__ == '__main__':
    unittest.main()
