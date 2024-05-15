
# April Ainsworth
# preparing for & running autoencoder

import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

audio_data = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/sensor_sound_merged.csv")

#sensor = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/hermiston_sensor_df.csv")
#sound = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/hermiston_sensor_df.csv")

#print("Sensor: ", sensor.shape)
#print("Sound: ", sound.shape)


# narrowing down which columns contain the Hertz values, so I can get min/max of these columns as a whole
# the range 44:444 will include all, from C1_10 to C3991_4000
# everything row 23 onward, and columns C2000_2010 to C_3991_4000 [:, 244:444] is NULL
# need to subset to this range in order to scale before processing the autoencoder

# column 244 onward is NULL, need to remove at least for now (see note in 1-hermiston-sound-preprocessing.py)
# print(audio_data.iloc[30:36,244:444])

non_hertz_subset_1 = audio_data.iloc[:, 0:44]
#print("Non Hertz 1:", non_hertz_subset_1)
non_hertz_subset_2 = audio_data.iloc[:, 444:451]
#print("Non Hertz 2:", non_hertz_subset_2)
hertz_subset = audio_data.iloc[:, 44:244]

# values of Hertz bins are right-skewed, needs log scaling
# hertz_dist = hertz_subset.iloc[:,0:10]
# hertz_dist.hist(alpha=0.5, figsize=(10, 8))
# plt.title('Hertz Distributions')
# plt.show()


nan_counts = non_hertz_subset_1.isna().sum()
print("NaN counts:")
print(nan_counts)

min_value = hertz_subset.min().min()
max_value = hertz_subset.max().max()

print(f"Min: ", min_value, "Max: ", max_value, "Total Length: ", len(audio_data)) # results are min 5, max 86312

# scaling data to fit the min/max range (absolute maximum scaling) # is Hertz data skewed post-minmax?
hertz_columns = audio_data.columns[44:244]
hertz_subset = audio_data[hertz_columns].astype(float) / 86312.

# instead use log scaling
# epsilon = 1e-9
# hertz_subset_log = np.log(hertz_subset + epsilon)
# hertz_subset_log = pd.DataFrame(hertz_subset_log, columns=hertz_subset_log.columns)

first_ten = hertz_subset.iloc[:,0:10]
first_ten.hist(bins=50, figsize=(20, 15))
plt.show()

# Check for NaN values in the entire DataFrame
#nan_df = audio_data_subset.isna()
#print(nan_df)

# Count the number of NaN values in each column
#nan_count_per_column = audio_data_subset.isna().sum()
#print(nan_count_per_column)

# Check if any NaN values exist in the DataFrame
#any_nan = audio_data_subset.isna().any().any()
#print("Any NaN values:", any_nan)

# print(audio_data.iloc[1:6, 43:444])
#print(audio_data.shape) # shape is (484608, 450)
#print(audio_data_subset.shape)
#print(audio_data_subset.iloc[0:20, 0:10]) # getting a slice to screenshot for slides
# print(hertz_columns)
      
kf = KFold(n_splits=5)

mse_scores = []
full_reconstructed_data = np.array([]).reshape(0,200)

X = hertz_subset

# Iterate over folds
for train_index, test_index in kf.split(hertz_subset):
    X_train, X_test = hertz_subset.iloc[train_index], hertz_subset.iloc[test_index]
    
    input_data = keras.Input(shape=(200,))

    # Define and compile autoencoder
    #autoencoder = keras.Sequential([
    #    # layers.Dense(100, activation='relu', input_shape=(200,)),
    #    layers.Dense(100, activation='relu')(input_data),
    #    layers.Dense(200, activation='sigmoid')
    #])

    encoder_output = layers.Dense(50, activation='relu')(input_data)
    decoder_output = layers.Dense(200, activation='sigmoid')(encoder_output)
    
    # Combine encoder and decoder into a single model
    autoencoder = keras.Model(input_data, decoder_output)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train autoencoder
    autoencoder.fit(X_train, X_train, epochs=1, batch_size=32, verbose=0)

    # Evaluate on test data
    reconstructed_data = autoencoder.predict(X_test)

    if full_reconstructed_data.size == 0:
        full_reconstructed_data = reconstructed_data
    else:
        full_reconstructed_data = np.vstack((full_reconstructed_data, reconstructed_data))

    # Compute MSE
    mse = mean_squared_error(X_test, reconstructed_data)
    mse_scores.append(mse)

# Print MSE for each fold
for i, mse in enumerate(mse_scores):
    print(f"Fold {i+1} Mean Squared Error (MSE): {mse}")

# Calculate and print average MSE across all folds
avg_mse = sum(mse_scores) / len(mse_scores)
print("Average Mean Squared Error (MSE) across all folds:", avg_mse)

# pre- and post-autoencoder
print(hertz_subset.iloc[0:9, 0:20])

full_reconstructed_df = pd.DataFrame(full_reconstructed_data, index=hertz_subset.index)
print(full_reconstructed_df.iloc[0:9, 0:20])
print(full_reconstructed_df.shape)

# reconstructed_data.to_csv("post-autoencoded-reconstruction.csv")

# plot mean squared error per fold
# NOTES: MSE is very small but fluctuates from run to run

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(mse_scores)+1), mse_scores, marker='o', linestyle='-', color='b')
plt.title('Mean Squared Error By Fold')
plt.xlabel('Fold Number')
plt.ylabel('MSE')
plt.grid(True)
plt.xticks(range(1, len(mse_scores)+1))
plt.show()

# save the hertz-only subset
full_reconstructed_df.to_csv("audio_data_subset_post_encoder.csv")

# paste together the whole dataset and save
non_hertz_subset_1_df = pd.DataFrame(non_hertz_subset_1, index=hertz_subset.index)
non_hertz_subset_2_df = pd.DataFrame(non_hertz_subset_2, index=hertz_subset.index)

audio_data_post_encoder_combined = pd.concat([non_hertz_subset_1_df, full_reconstructed_df, non_hertz_subset_2_df], axis=1, ignore_index=True)
audio_data_post_encoder_combined.to_csv("audio_data_post_encoder_combined.csv")