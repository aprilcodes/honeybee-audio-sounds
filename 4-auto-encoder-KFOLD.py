
# April Ainsworth
# preparing for & running autoencoder

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
from keras import layers

audio_data = pd.read_csv("C:/ncf-graduate-school/internship-USDA/almond-pollination/data/sensor_sound_merged.csv")

# narrowing down which columns contain the Hertz values, so I can get min/max of these columns as a whole
# the range 44:444 will include all, from C1_10 to C3991_4000
# everything row 23 onward, and columns C2000_2010 to C_3991_4000 [:, 244:444] is NULL
# need to subset to this range in order to scale before processing the autoencoder

# column 244 onward is NULL, need to remove at least for now (see note in 1-hermiston-sound-preprocessing.py)
# print(audio_data.iloc[30:36,244:444])

non_hertz_subset_1 = audio_data.iloc[:, 0:44]
print("Non Hertz 1:", non_hertz_subset_1)

non_hertz_subset_2 = audio_data.iloc[:, 444:451]
print("Non Hertz 2:", non_hertz_subset_2)

hertz_subset = audio_data.iloc[:, 44:244]
min_value = hertz_subset.min().min()
max_value = hertz_subset.max().max()

print(f"Min: ", min_value, "Max: ", max_value, "Total Length: ", len(audio_data)) # results are min 5, max 86312

# scaling data to fit the min/max range (absolute maximum scaling)

hertz_columns = audio_data.columns[44:244]
audio_data_subset = audio_data[hertz_columns].astype(float) / 86312.

# Check for NaN values in the entire DataFrame
nan_df = audio_data_subset.isna()
print(nan_df)

# Count the number of NaN values in each column
nan_count_per_column = audio_data_subset.isna().sum()
print(nan_count_per_column)

# Check if any NaN values exist in the DataFrame
any_nan = audio_data_subset.isna().any().any()
print("Any NaN values:", any_nan)

# print(audio_data.iloc[1:6, 43:444])
# print(audio_data.shape) # shape is (484608, 450)

kf = KFold(n_splits=5)

mse_scores = []

X = audio_data_subset

# Iterate over folds
for train_index, test_index in kf.split(audio_data_subset):
    X_train, X_test = audio_data_subset.iloc[train_index], audio_data_subset.iloc[test_index]
    
    input_data = keras.Input(shape=(200,))

    # Define and compile autoencoder
    #autoencoder = keras.Sequential([
    #    # layers.Dense(100, activation='relu', input_shape=(200,)),
    #    layers.Dense(100, activation='relu')(input_data),
    #    layers.Dense(200, activation='sigmoid')
    #])

    encoder_output = layers.Dense(100, activation='relu')(input_data)
    decoder_output = layers.Dense(200, activation='sigmoid')(encoder_output)
    
    # Combine encoder and decoder into a single model
    autoencoder = keras.Model(input_data, decoder_output)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train autoencoder
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate on test data
    reconstructed_data = autoencoder.predict(X_test)

    # Compute MSE
    mse = mean_squared_error(X_test, reconstructed_data)
    mse_scores.append(mse)

# Print MSE for each fold
for i, mse in enumerate(mse_scores):
    print(f"Fold {i+1} Mean Squared Error (MSE): {mse}")

# Calculate and print average MSE across all folds
avg_mse = sum(mse_scores) / len(mse_scores)
print("Average Mean Squared Error (MSE) across all folds:", avg_mse)