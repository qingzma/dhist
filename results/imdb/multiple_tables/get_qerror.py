import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from joins.tools import read_from_csv

root_path = "/home/lrr/Documents/research/card/"


def q_error(truths, estimations):
    q_errors = np.array([])
    assert len(truths) == len(estimations)
    for i in range(len(truths)):
        # if estimations[i] == 1:
        #     continue
        if estimations[i] == 0:
            estimations[i] = 1
            # continue
        error = truths[i] / estimations[i]
        if error < 1:
            q_errors = np.append(q_errors, estimations[i] / truths[i])
            # print(i, estimations[i] / truths[i])
        else:
            q_errors = np.append(q_errors, error)
            # print(i, error)

    median_error = np.median(q_errors)
    percentile_90 = np.percentile(q_errors, 90)
    percentile_95 = np.percentile(q_errors, 95)
    percentile_99 = np.percentile(q_errors, 99)
    permax = max(q_errors)
    print('50%', median_error)
    print('90%', percentile_90)
    print('95%', percentile_95)
    print('99%', percentile_99)
    print('max', permax)
    print('--------------------------')
    return q_errors


def plot_qerror():
    truths = read_from_csv(root_path + "results/imdb/multiple_tables/truth_sub_queries.csv", "est")
    # card = read_from_csv(root_path + "results/imdb/multiple_tables/card_sub_queries_200_20.csv", "card")
    # card = read_from_csv(root_path + "results/imdb/multiple_tables/card_job_light_200_20.csv", "card")
    card = read_from_csv("/home/lrr/Documents/research/card/results/imdb/multiple_tables/card_test_binsize_0.07.csv", "card")
    postgres = read_from_csv(root_path + "results/imdb/multiple_tables/postgres.csv", "postgres")
    oracle = read_from_csv(root_path + "results/imdb/multiple_tables/oracle.csv", "oracle")
    # deepdb = read_from_csv(root_path + "results/imdb/multiple_tables/imdb_light_model_based_budget_5.csv", "cardinality_predict")
    deepdb = read_from_csv(root_path + "results/imdb/multiple_tables/joblight_deepdb_subs.csv",
                           "cardinality_predict")

    # factorjoin = read_from_csv(root_path + "results/imdb/multiple_tables/job_light_sub_queries_factorjoin.csv",
    #                            "factorjoin")
    factorjoin = read_from_csv(root_path + "results/imdb/multiple_tables/factorjoin.csv",
                               "factorjoin")
    # wjsample = read_from_csv(root_path + "results/imdb/multiple_tables/job_light_sub_queries_wjsample.csv", "wjsample")
    wjsample = read_from_csv(root_path + "results/imdb/multiple_tables/wjsample_1115.csv", "wjsample")
    bayescard = read_from_csv(root_path + "results/imdb/multiple_tables/bayescard_gived.csv", "bayescard")
    flat = read_from_csv(root_path + "results/imdb/multiple_tables/flat_gived.csv", "flat")

    truth_joblight = read_from_csv(root_path + "results/imdb/multiple_tables/truth.csv", "est")
    card_joblight = read_from_csv(root_path + "results/imdb/multiple_tables/card_job_light_200_20.csv", "card")
    deepdb_joblight = read_from_csv(
        "/home/lrr/Documents/research/deepdb-public/baselines/cardinality_estimation/results/deepDB/imdb_light_model_based_budget_10.csv",
        'cardinality_predict')
    # q_card = q_error(truths, card)
    # q_postgres = q_error(truths, postgres)
    # q_oracle = q_error(truths, oracle)
    q_wjsample = q_error(truths, wjsample)
    # q_deepdb = q_error(truths, deepdb)
    # q_factorjoin = q_error(truths, factorjoin)
    # q_bayescard = q_error(truths, bayescard)
    # q_flat = q_error(truths, flat)

    # q_deepdb = q_error(truth_joblight, deepdb_joblight)
    # print(q_deepdb)
    
    
def plot_latency():
    card = read_from_csv(root_path + "results/imdb/multiple_tables/card_job_light_200_20.csv", "card-time")
    # factorjoin = read_from_csv(
    #     root_path + "results/imdb/multiple_tables/factorjoin.csv", "factorjoin-time"
    # )
    # flat = read_from_csv(
    #     root_path + "workloads/imdb_CEB/estimates/imdb_CEB_sub_queries_flat.txt", "flat"
    # )
    # bayescard = read_from_csv(
    #     root_path + "workloads/imdb_CEB/estimates/imdb_CEB_sub_queries_bayescard.txt", "bayescard"
    # )
    wjsample = read_from_csv(
        root_path + "results/imdb/multiple_tables/wjsample_1115.csv", "wjsample-time")
    deepdb = read_from_csv(
        root_path + "results/imdb/multiple_tables/joblight_deepdb_subs.csv", "latency_ms"
    )
    oracle = read_from_csv(
        "/home/lrr/Documents/research/card/results/imdb/multiple_tables/oracle.csv", "exec_time"
    )
    postgres = read_from_csv(
        root_path + "results/imdb/multiple_tables/postgres.csv", "postgres-time")

    card *= 1000
    postgres *= 1000
    oracle *= 1000
    # factorjoin *= 1000

    print("average of card is ", np.mean(card))
    print("average of postgres is ", np.mean(postgres))
    print("average of oracle is ", np.mean(oracle))
    print("average of wjsample is ", np.mean(wjsample))
    print("average of deepdb is ", np.mean(deepdb))
    # print("average of factorjoin is ", np.mean(factorjoin))
    # print([truths != -1][0])
    
    
if __name__ == '__main__':
    plot_qerror()
    plot_latency()



    # df1 = pd.read_csv('/home/lrr/Documents/research/card/results/imdb/multiple_tables/card_sub_queries_200_20.csv')['card']
    # df2 = pd.read_csv('/home/lrr/Documents/research/card/results/imdb/multiple_tables/truth_sub_queries.csv')['est']
    # df_bayes = pd.read_csv('/home/lrr/Documents/End-to-End-CardEst-Benchmark/workloads/job-light/sub_plan_queries/estimates/job_light_sub_queries_neurocard.txt',
    #                        header=None)[0]
    # df_postgres = pd.read_csv('/home/lrr/Documents/research/card/results/imdb/multiple_tables/postgres.csv',)['postgres']
    # print(len(df_bayes))
    # print(df_bayes.head())
    # errors = []
    # for i in range(len(df1)):
    #     truth = float(df2[i][2:-3])
    #     est = float(df1[i])
    #     # est = float(df_bayes[i])
    #     if est == 0.0:
    #         print('--')
    #         est = 1.0
    #     error = max((est / truth), (truth / est))
    #     # error = est / truth
    #     errors.append(error)
    # median_error = np.median(errors)
    # percentile_90 = np.percentile(errors, 90)
    # percentile_95 = np.percentile(errors, 95)
    # # percentile_99 = np.percentile(errors, 99)
    # permax = max(errors)
    # print(median_error)
    # print(percentile_90)
    # print(percentile_95)
    # # print(percentile_99)
    # print(permax)
    #
    # # logbins = 301
    # plt.xscale("log")
    # # plt.xticks([1, 10, 100, 1000, 10000,])
    # plt.hist(errors, label="DHist", bins=10000)
    # plt.yticks()

    # for ax in axs:
    #     for a in ax:
    #         a.set_yscale("log")
    #         a.set_xscale("log")
    #         a.set_ylim([0.1, 1000])
    #         a.set_xticks([0.0001, 0.001, 0.01, 0.1, 1,
    #                       10, 100, 1000, 10000, 100000])
    #         a.tick_params(axis='x', labelsize=8)  # 调整 x 轴字体大小
    #         a.tick_params(axis='y', labelsize=8)
    #
    # fig.text(0.5, 0.01, "Relative error", ha="center")
    # fig.text(0.01, 0.5, "Number of queries", va="center", rotation="vertical")
    # plt.tight_layout()
    # plt.show()