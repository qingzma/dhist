import os
import time

import pandas as pd


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def convert_time_to_int(data_folder):
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            csv_file_location = data_folder + file
            df_rows = pd.read_csv(csv_file_location)
            for attribute in df_rows.columns:
                if "Date" in attribute:
                    if df_rows[attribute].values.dtype == "object":
                        new_value = []
                        for value in df_rows[attribute].values:
                            new_value.append(timestamp_transorform(value))
                        df_rows[attribute] = new_value
            df_rows.to_csv(csv_file_location, index=False)


def q_error(res, true_card):
    return max(res / true_card, true_card / res)


def rel_error(pred, truth):
    return pred / truth


def save_predictions_to_file(preds, times, header1, header2, file_path):
    data = zip(preds, times)
    data = list(data)
    df = pd.DataFrame(data, columns=[header1, header2])
    df.to_csv(file_path, index=False)


def read_from_csv(file_path, header):
    return pd.read_csv(file_path)[header].values
