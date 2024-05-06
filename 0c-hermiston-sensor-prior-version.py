import re
import pandas as pd
import requests

url = "https://hw.hermistonconsulting.com/sensordata_25.sql"
response = requests.get(url)

if response.status_code == 200:
    sql_file_content = response.text

# sql_file_content = open("C:/ncf-graduate-school/internship-USDA/sensordata_25.sql")

# with open("C:/ncf-graduate-school/internship-USDA/sensordata_25.sql", "r") as file:
#     sql_file_content = file.read()

pattern = r'\(([^)]*?)\)'
all_rows = re.findall(pattern, sql_file_content)

data = []

# parsing each row
for line in all_rows:
    values = line.split(',')
    cleaned_values = [v.strip("'").strip() for v in values]
    data.append(cleaned_values)

# make a dataframe from the rows
df = pd.DataFrame(data, columns=['id', 'sensor_id', 'timestamp', 'voltage', 'battery_voltage', 'modem_rssi', 'temperature', 'pcb_temperature_one', 'pcb_temperature_two', 'firmware_version', 'acceleration_x', 'acceleration_y', 'acceleration_z', 'magnetic_x', 'magnetic_y', 'magnetic_z', 'light_ir', 'light_blue', 'light_red', 'external_humidity', 'external_temperature', 'external_weight', 'light_green', 'modem_voltage', 'gateway_id', 'config', 'gateway_mac_address', 'sensor_mac_address', 'humidity', 'blerssi', 'in_count', 'out_count', 'group_id', 'is_installed', 'hive_id', 'upload_time', 'phone_id', 'created', 'light'])

in_count = df.iloc[:, 30]
out_count = df.iloc[:, 31]

# remove any version of a null/nan
in_count_non_null = in_count[(in_count != None) & (in_count != "None") & (in_count != "NULL") & pd.notna(in_count)]

print(in_count_non_null) # we get 3455 occurrences of something that's not null, but they are just repeats of the INSERT line seen periodically in the data

# another way to get a "window" into the data to view this... see that `in_count` and `out_count` are just placeholders
print(df.iloc[15:20, 28:34])

# checking what's left if we also remove `in_count`
is_anything_left = in_count[(in_count != None) & (in_count != "None") & (in_count != "NULL") & pd.notna(in_count) & (in_count != "`in_count`")]
print("After removing nulls and placeholders: ", is_anything_left)