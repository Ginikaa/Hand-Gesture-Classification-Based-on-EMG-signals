import numpy as np
import os
import csv
import pandas as pd
import math
from scipy.signal import medfilt, butter, lfilter, filtfilt
import matplotlib.pyplot as plt

file_path = "./Project_Data_EE4C12_S&S_EMG.csv"
raw_data = pd.read_csv(file_path)

grouped = raw_data.groupby('label')

field = ['RMS1', 'VAR1', 'MAV1', 'SSC1', 'WL1', 'RMS2', 'VAR2', 'MAV2', 'SSC2', 'WL2',
         'RMS3', 'VAR3', 'MAV3', 'SSC3', 'WL3', 'RMS4', 'VAR4', 'MAV4', 'SSC4', 'WL4',
         'RMS5', 'VAR5', 'MAV5', 'SSC5', 'WL5', 'RMS6', 'VAR6', 'MAV6', 'SSC6', 'WL6',
         'RMS7', 'VAR7', 'MAV7', 'SSC7', 'WL7', 'RMS8', 'VAR8', 'MAV8', 'SSC8', 'WL8', 'class', 'label']

#  All the features come from the reference
#  "Electromyogram-Based Classification of Hand and Finger Gestures Using Artificial Neural Networks"
#  RMS:Root mean square
#  VAR:Variance
#  MAV:Mean absolute value
#  SSC:Slop sign change
#  WL:Waveform length

file = open("D:/PY3/EMGProject/emg_feature.csv", 'w', newline='')
writer_feature = csv.writer(file)
writer_feature.writerow(field)


def rms(signal):
    squared_data = [x ** 2 for x in signal]
    sum_of_squares = sum(squared_data)

    result = math.sqrt(sum_of_squares / len(signal))
    return result


def calculate_ssc(signal):
    ssc_count = 0
    prev_slope = 0

    for j in range(1, len(signal)):
        current_slope = signal[j] - signal[j - 1]

        if current_slope == 0:
            pass
        elif current_slope * prev_slope < 0:
            ssc_count += 1  # The sign of the slope changes

        prev_slope = current_slope

    return ssc_count


def calculate_waveform_length(data):
    wl = 0

    for k in range(1, len(data)):
        diff = abs(data[k] - data[k - 1])
        wl += diff

    return wl


def raw_data_processing(df):
    # Complete data (fill in missing time points)
    expected_time_interval = 1  # Since we assume the sample frequency is 1000 Hz
    expected_time_sequence = range(df['time'].min(), df['time'].max() + 1, expected_time_interval)
    set1 = set(expected_time_sequence)
    # missing_time_points = [time for time in expected_time_sequence if time not in df['time']]
    set2 = set(df['time'])
    unique_elements = list(set1.symmetric_difference(set2))

    missing_time_points = list(unique_elements)
    missing_data = pd.DataFrame({'time': missing_time_points})

    # Merge original data and filled missing data
    result_df = pd.concat([df, missing_data]).sort_values(by=['time'])
    result_df = result_df.fillna(method='ffill')
    result_df.reset_index(drop=True, inplace=True)

    df_array = np.array(result_df)
    iter_array = df_array.copy()

    for channel in range(1, 9):
        # band-pass Butterworth filter at 10-400Hz, and sample frequency is 1000Hz
        b, a = butter(4, [10 / 500, 400 / 500], btype='band', analog=False)
        filtered = filtfilt(b, a, df_array[:, channel])
        iter_array[:, channel] = filtered.copy()
        result_df.iloc[:, channel] = filtered.copy()

        # filtered_signal = medfilt(df_array[:, channel], kernel_size=51)
        # iter_array[:, channel] = filtered_signal.copy()
        # divisible_by_5_df.iloc[:, channel] = filtered_signal.copy()

        fig, ax = plt.subplots( figsize=(12, 8))
        ax.plot(result_df['time'], df_array[:, channel],'b-', label='raw signal')
        ax.plot(result_df['time'], filtered,'r-', label='filtered signal')
        ax.legend()
        plt.show()

    array_filtered = iter_array.copy()
    df_filtered = pd.DataFrame(result_df)

    return df_filtered, array_filtered


def feature_extraction(filter_arr):
    start = 0
    array = filter_arr[filter_arr[:, -2] != 0]    # ignore Class 0
    for idx in range(1, array.shape[0]):
        # if array[idx - 1, -2] != 0:
        #     continue

        if array[idx, -2] != array[idx - 1, -2]:    # slide the windows "within" the class
            array_class = array[start: idx, :]
            start = idx

            # The length of each window is 250 ms with interval 25 ms.
            # We can ignore the time column and the segmentation can be based on the Class.
            # And they are overlap windows.
            num_windows = array_class.shape[0] // 275
            windows = np.arange(0, num_windows)
            for window in windows:
                row = []
                window_signal = array_class[window * 25: (window * 25 + 250), :]
                for column in range(1, 9):
                    channel_rms = rms(window_signal[:, column])
                    channel_var = np.var(window_signal[:, column])
                    channel_mav = np.mean(np.abs(window_signal[:, column]))
                    channel_ssc = calculate_ssc(window_signal[:, column])
                    channel_wl = calculate_waveform_length(window_signal[:, column])
                    row.append(channel_rms)
                    row.append(channel_var)
                    row.append(channel_mav)
                    row.append(channel_ssc)
                    row.append(channel_wl)
                row.append(window_signal[0, -2])
                row.append(window_signal[0, -1])
                writer_feature.writerow(row)


for name, group in grouped:
    for i in range(len(group['time'])):
        if group.iloc[i + 1, 0] < group.iloc[i, 0]:
            trail1 = group.iloc[:i + 1, :]
            trail2 = group.iloc[i + 1:, :]
            break
    filtered_df1, filtered_array1 = raw_data_processing(trail1)
    filtered_df2, filtered_array2 = raw_data_processing(trail2)
    output_path1 = os.path.join("D:/PY3/EMGProject/filtered", f"P{name}_trail1_filtered.csv")
    output_path2 = os.path.join("D:/PY3/EMGProject/filtered", f"P{name}_trail2_filtered.csv")
    filtered_df1.to_csv(output_path1)
    filtered_df2.to_csv(output_path2)
    feature_extraction(filtered_array1)
    feature_extraction(filtered_array2)
