# April Ainsworth
# 4/27/2024

# this code formats the sql file sound_25.sql into a pandas df,
# removes the first few lines that are empty rows, 
# then saves it to a csv file

import re
import pandas as pd

with open("C:/ncf-graduate-school/internship-USDA/sensordata_25.sql", "r") as file:
     sensor_data = file.read()

pattern = r'\((.*?)\)'
all_rows = re.findall(pattern, sensor_data) # all_rows is a list

data = []

# parsing each row
for line in all_rows:
    values = line.split(',')
    cleaned_values = [v.strip("'").strip() for v in values]
    data.append(cleaned_values)

df = pd.DataFrame(data, columns=['id', 'sensor_id', 'timestamp', 'voltage', 'battery_voltage', 'modem_rssi', 'temperature', 'pcb_temperature_one', 'pcb_temperature_two', 'firmware_version', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'magnetic_x', 'magnetic_y', 'magnetic_z', 'light_ir', 'light_blue', 'light_red', 'external_humidity', 'external_temperature', 'external_weight', 'light_green', 'modem_voltage', 'gateway_id', 'config', 'gateway_mac_address', 'sensor_mac_address', 'humidity', 'blerssi', 'in_count', 'out_count', 'group_id', 'is_installed', 'hive_id', 'upload_time', 'phone_id', 'created', 'light'])

# rows 0:16 are empty due to SQL formatting, last 3 rows also empty
df = df[17:-3]

# filtering out the periodic rows that appear with headers in them 
# (the tilde means NOT contains)

df['id'] = df['id'].astype(str)
filtered_df = df[~df['id'].str.contains('id')]

filtered_df.to_csv("hermiston_sensor_df.csv")