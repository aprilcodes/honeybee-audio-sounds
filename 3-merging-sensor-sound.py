# April Ainsworth
# 04/27/2024

# merges two files based on timestamp with a fuzzy match
# cuts down data to only timestamps containing data in pertinent columns

import pandas as pd

sensor = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/hermiston_sensor_df.csv")
sound = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/hermiston_sound_df.csv")

#print(sensor[140:145]) # sensor data still has headers in it
#print("----------------------------------")
#print(sound[140:145])

# nan_df = sensor.isna()
# print(f"SENSOR NANS: ", nan_df)

# Count the number of NaN values in each column
# nan_count_per_column = sensor.isna().sum()
# print(nan_count_per_column)

# Check if any NaN values exist in the DataFrame
# any_nan = sensor.isna().any().any()
# print("SENSOR NaN values:", any_nan)

# nan_df = sound.isna()
# print(f"SOUND NANS: ", nan_df)

# Count the number of NaN values in each column
# nan_count_per_column = sound.iloc[:, 345:350].isna().sum()
# print(nan_count_per_column)

# Check if any NaN values exist in the DataFrame
# any_nan = sound.isna().any().any()
# print("SOUND NaN values:", any_nan)

sensor['timestamp'] = pd.to_datetime(sensor['timestamp'])
sound['timestamp'] = pd.to_datetime(sound['timestamp'])

sensor.sort_values(by = 'timestamp', inplace = True)
sound.sort_values(by = 'timestamp', inplace = True)

# pd.merge_asof provides fuzzy match
sensor_sound_merged = pd.merge_asof(sensor, sound, on = "timestamp")

# sensor data runs 11/2/23 to 4/5/24; sound data runs 10/22/23 to 3/8/24
# this eliminates rows not common to both (original merge: 487366 rows, post NaNs removed on C1_10: 484,608)
sensor_sound_merged = sensor_sound_merged[~pd.isna(sensor_sound_merged['C1_10'])]

sensor_sound_merged.to_csv("sensor_sound_merged.csv")