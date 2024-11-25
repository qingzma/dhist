import pandas as pd
import numpy as np
from joins.tools import read_from_csv_all


def get_exec_time(csv_path):
    exec_time = read_from_csv_all(csv_path)['execution-time']
    return exec_time


def get_plan_time(csv_path):
    plan_time = read_from_csv_all(csv_path)['plan-time']
    return plan_time


def get_timeout(csv_path):
    exec_true = read_from_csv_all(csv_path)['truth']
    nums = 0
    for i in exec_true:
        if i == -1:
            nums += 1
    return nums


if __name__ == '__main__':
    # root_path = '/home/lrr/Documents/research/card/results/stats/end_to_end/without_timeout'
    root_path = '/home/lrr/Documents/research/card/results/stats/end_to_end/with_timeout'
    # root_path = '/home/lrr/Documents/research/card/results/stats/end_to_end/machine'
    models = [
        # 'truth',
        # 'postgres',
        # 'oracle',
        # 'wjsample',
        # 'neurocard',
        # 'flat',
        'deepdb',
        'bayescard',
        'factorjoin',
        'dhist'
    ]
    for model in models:
        print('------------')
        exec_time = get_exec_time(root_path + '/' + model + '.csv')
        plan_time = get_plan_time(root_path + '/' + model + '.csv')
        # timeouts = get_timeout(root_path + '/' + model + '.csv')

        print(model + ' exec_time(s):', np.sum(exec_time) / 1000)
        print(model + ' plan_time(s):', np.sum(plan_time) / 1000)
        print(model + ' timeout:', get_timeout(root_path + '/' + model + '.csv'))
