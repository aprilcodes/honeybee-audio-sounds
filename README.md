# honeybee-audio-sounds
Machine learning project on honeybee audio data

Sensor & Sound Data Files Pre-Processing & Modeling

1 & 2:
- converted SQL file into a Pandas dataframe
- removed periodic header rows
- removed leading/trailing empty rows
- saved as csv files: hermiston_sensor_df, hermiston_sound_df

3: merged sensor and sound on timestamps with fuzzy match
- saved as sensor_sound_merged.csv

4: realized that > 2000 Hz values are not all available, cut those
- scaled data
- ran an auto-encoder to 100 dimensions with 5-fold cross validation
