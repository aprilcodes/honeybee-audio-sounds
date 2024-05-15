
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

audio_data = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/audio_data_post_encoder_combined.csv")

# in audio_data, columns are:
# 0 to 43 are non-hertz cols
# 3: sensor_id
# 4: timestamp
# 9: pcb_temperature_one
# 30: humidity
# 44 to 243 are hertz cols: C1_10 to C3991_4000
# 244: centroid
# 245: volume

# print(audio_data.iloc[0:3, 44:54])

# cut the columns not used in modeling
ranges_to_drop = [range(0, 4), 
                  range(6, 10),  
                  range(11, 31),
                  range(32, 45),  # Columns 31 through 43
                  range(247, len(audio_data.columns))]  # Columns 246 to the last column

columns_to_drop = [column for r in ranges_to_drop for column in r]
# print(columns_to_drop)

audio_UMAP_input = audio_data.drop(audio_data.columns[columns_to_drop], axis=1)

audio_UMAP_input.columns = (['sensor_id', 'timestamp', 'pcb_temperature_one', 'humidity', 'C1_10', 'C11_20', 'C21_30', 'C31_40', 'C41_50', 'C51_60', 'C61_70', 'C71_80', 'C81_90', 'C91_100', 'C101_110', 'C111_120', 'C121_130', 'C131_140', 'C141_150', 'C151_160', 'C161_170', 'C171_180', 'C181_190', 'C191_200', 'C201_210', 'C211_220', 'C221_230', 'C231_240', 'C241_250', 'C251_260', 'C261_270', 'C271_280', 'C281_290', 'C291_300', 'C301_310', 'C311_320', 'C321_330', 'C331_340', 'C341_350', 'C351_360', 'C361_370', 'C371_380', 'C381_390', 'C391_400', 'C401_410', 'C411_420', 'C421_430', 'C431_440', 'C441_450', 'C451_460', 'C461_470', 'C471_480', 'C481_490', 'C491_500', 'C501_510', 'C511_520', 'C521_530', 'C531_540', 'C541_550', 'C551_560', 'C561_570', 'C571_580', 'C581_590', 'C591_600', 'C601_610', 'C611_620', 'C621_630', 'C631_640', 'C641_650', 'C651_660', 'C661_670', 'C671_680', 'C681_690', 'C691_700', 'C701_710', 'C711_720', 'C721_730', 'C731_740', 'C741_750', 'C751_760', 'C761_770', 'C771_780', 'C781_790', 'C791_800', 'C801_810', 'C811_820', 'C821_830', 'C831_840', 'C841_850', 'C851_860', 'C861_870', 'C871_880', 'C881_890', 'C891_900', 'C901_910', 'C911_920', 'C921_930', 'C931_940', 'C941_950', 'C951_960', 'C961_970', 'C971_980', 'C981_990', 'C991_1000', 'C1001_1010', 'C1011_1020', 'C1021_1030', 'C1031_1040', 'C1041_1050', 'C1051_1060', 'C1061_1070', 'C1071_1080', 'C1081_1090', 'C1091_1100', 'C1101_1110', 'C1111_1120', 'C1121_1130', 'C1131_1140', 'C1141_1150', 'C1151_1160', 'C1161_1170', 'C1171_1180', 'C1181_1190', 'C1191_1200', 'C1201_1210', 'C1211_1220', 'C1221_1230', 'C1231_1240', 'C1241_1250', 'C1251_1260', 'C1261_1270', 'C1271_1280', 'C1281_1290', 'C1291_1300', 'C1301_1310', 'C1311_1320', 'C1321_1330', 'C1331_1340', 'C1341_1350', 'C1351_1360', 'C1361_1370', 'C1371_1380', 'C1381_1390', 'C1391_1400', 'C1401_1410', 'C1411_1420', 'C1421_1430', 'C1431_1440', 'C1441_1450', 'C1451_1460', 'C1461_1470', 'C1471_1480', 'C1481_1490', 'C1491_1500', 'C1501_1510', 'C1511_1520', 'C1521_1530', 'C1531_1540', 'C1541_1550', 'C1551_1560', 'C1561_1570', 'C1571_1580', 'C1581_1590', 'C1591_1600', 'C1601_1610', 'C1611_1620', 'C1621_1630', 'C1631_1640', 'C1641_1650', 'C1651_1660', 'C1661_1670', 'C1671_1680', 'C1681_1690', 'C1691_1700', 'C1701_1710', 'C1711_1720', 'C1721_1730', 'C1731_1740', 'C1741_1750', 'C1751_1760', 'C1761_1770', 'C1771_1780', 'C1781_1790', 'C1791_1800', 'C1801_1810', 'C1811_1820', 'C1821_1830', 'C1831_1840', 'C1841_1850', 'C1851_1860', 'C1861_1870', 'C1871_1880', 'C1881_1890', 'C1891_1900', 'C1901_1910', 'C1911_1920', 'C1921_1930', 'C1931_1940', 'C1941_1950', 'C1951_1960', 'C1961_1970', 'C1971_1980', 'C1981_1990', 'C1991_2000', 'centroid', 'volume'])

# did thorough comparisons before/after to ensure no necessary columns were accidentally dropped

# TBD: is sensor_id useful to the modeling at all? does it give us info that's specific to the hive or its activities?
# same question for volume... we have amplitude in the cells, so does volume provide anything additional?
# drop both for now
audio_UMAP_input = audio_UMAP_input.drop(audio_UMAP_input.columns[0], axis=1)
audio_UMAP_input = audio_UMAP_input.drop('volume', axis=1)

# print(audio_UMAP_input.iloc[0:3, 0:9])

# there are some zero values in humidity and centroid; replacing zeros with mean of each respective column
mean_centroid = audio_UMAP_input.loc[audio_UMAP_input['centroid'] != 0, 'centroid'].mean()
mean_humidity = audio_UMAP_input.loc[audio_UMAP_input['humidity'] != 0, 'humidity'].mean()

audio_UMAP_input.loc[audio_UMAP_input['centroid'] == 0, 'centroid'] = mean_centroid
audio_UMAP_input.loc[audio_UMAP_input['humidity'] == 0, 'humidity'] = mean_humidity

# now that I'm employing timestamp for the first time, it needs to be numerical, not a string

audio_UMAP_input['timestamp'] = pd.to_datetime(audio_UMAP_input['timestamp'])
# audio_UMAP_input['timestamp'] = audio_UMAP_input['timestamp'].astype('int64') // 1_000_000_000 # scaling what might be very large #s
audio_UMAP_input['timestamp'] = audio_UMAP_input['timestamp'].astype('int64') # took the scaling off (from line above) bc we now scale below

# need to scale timestamp, humidity, centroid, pcb_temperature_one
scaler = MinMaxScaler()
columns_to_scale = ['timestamp', 'humidity', 'centroid', 'pcb_temperature_one']
audio_UMAP_input[columns_to_scale] = scaler.fit_transform(audio_UMAP_input[columns_to_scale])

# before standardization
last_ten = audio_UMAP_input.iloc[:,-10:]
last_ten.hist(alpha=0.5, bins = 50, figsize=(10, 8))
plt.title('Last Ten')
plt.show()

min_values_before = audio_UMAP_input.min()
max_values_before = audio_UMAP_input.max()
print("1st 10 MIN/MAX BEFORE STANDARDIZATION:----------")
print(min_values_before[0:10])
print("---------------")
print(max_values_before[0:10])

print("LAST 10 MIN/MAX BEFORE STANDARDIZATION:----------")
print(min_values_before[-10:])
print("---------------")
print(max_values_before[-10:])

# now to standardize (mean = 0, sd = 1) ### trialing the UMAP process while skipping standardization
# mean = audio_UMAP_input.mean()
# std = audio_UMAP_input.std()

# print("Mean Values: ", mean)
# print("Standard Deviation: ", std)

# Standardize the data
# audio_UMAP_input_standardized = (audio_UMAP_input - mean) / std

# min_values = audio_UMAP_input_standardized.min()
# max_values = audio_UMAP_input_standardized.max()
# print("MIN/MAX AFTER STANDARDIZATION:----------")
# print(min_values[0:10])
# print("---------------")
# print(max_values[0:10])

# after standardization
# first_ten_standardized = audio_UMAP_input.iloc[:,0:10]
# first_ten_standardized.hist(alpha=0.5, bins = 50, figsize=(10, 8))
# plt.title('After Standardization')
# plt.show()

# audio_UMAP_input_standardized.to_csv("audio_UMAP_input_standardized.csv")
audio_UMAP_input.to_csv("audio_UMAP_input.csv")
print("saved")