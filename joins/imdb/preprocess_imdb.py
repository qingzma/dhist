import pandas as pd


if __name__ == "__main__":
    headers = ['id', 'movie_id', 'info_type_id', 'info', 'note']
    df = pd.read_csv('/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/imdb/joblight/movie_info_idx.csv')
    # df.columns = headers
    print(df.tail())
    # df.to_csv('/home/lrr/Documents/End-to-End-CardEst-Benchmark/datasets/imdb/joblight/movie_info_idx.csv',
    #           index=False)
