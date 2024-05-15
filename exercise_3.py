import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import math as m
import shutil
from tfx_bsl.coders.example_coder import RecordBatchToExamplesEncoder
from tfx_bsl.public import tfxio

def visualize_plots(dataset, columns):
    features = dataset[columns]
    fig, axes = plt.subplots(
        nrows=len(columns)//2 + len(columns)%2, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i, col in enumerate(columns):
        c = colors[i % (len(colors))]
        t_data = dataset[col]
        t_data.index = dataset.index
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{}".format(col),
            rot=25,
        )
    ax.legend([col])
    plt.tight_layout()

def set_velocity_outliers(df):
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0
    return wv

def set_max_wind_vel(df):
    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0
    return bad_max_wv

def read_input_data(INPUT_FILE):
    return pd.read_csv(INPUT_FILE, header=0, index_col=0)

def show_correlation_heatmap(dataframe):
    plt.figure(figsize=(20,20))
    cor = dataframe.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
    plt.show()

def clean_fn(line):
  '''
  Converts datetime strings in the CSV to Unix timestamps and removes outliers
  the wind velocity column. Used as part of
  the transform pipeline.

  Args:
    line (string) - one row of a CSV file
  
  Returns:

  '''

  # Split the CSV string to a list
  line_split = line.split(b',')

  # Decodes the timestamp string to utf-8
  date_time_string = line_split[date_time_idx].decode("utf-8")

  # Creates a datetime object from the timestamp string
  date_time = datetime.strptime(date_time_string, '%d.%m.%Y %H:%M:%S')

  # Generates a timestamp from the object
  timestamp = datetime.timestamp(date_time)

  # Overwrites the string timestamp in the row with the timestamp in seconds
  line_split[date_time_idx] = bytes(str(timestamp), 'utf-8')

  # Check if wind velocity is an outlier
  if line_split[wv_idx] == b'-9999.0':

    # Overwrite with default value of 0
    line_split[wv_idx] = b'0.0'

  # rejoin the list item into one string
  mod_line = b','.join(line_split)

  return mod_line

def preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  
  outputs = inputs.copy()

  # Filter redundant features
  for key in FEATURES_TO_REMOVE:
    del outputs[key]

  # Convert degrees to radians
  pi = tf.constant(m.pi)
  wd_rad = inputs['wd (deg)'] * pi / 180.0

  # Calculate the wind x and y components.
  outputs['Wx'] = inputs['wv (m/s)'] * tf.math.cos(wd_rad)
  outputs['Wy'] = inputs['wv (m/s)'] * tf.math.sin(wd_rad)

  # Delete `wv (m/s)` and `wd (deg)` after getting the wind vector
  del outputs['wv (m/s)']
  del outputs['wd (deg)']

  # Get day and year in seconds
  day = tf.cast(24*60*60, tf.float32)
  year = tf.cast((365.2425)*day, tf.float32)

  # Get timestamp feature
  timestamp_s = outputs['Date Time']

  # Convert timestamps into periodic signals
  outputs['Day sin'] = tf.math.sin(timestamp_s * (2 * pi / day))
  outputs['Day cos'] = tf.math.cos(timestamp_s * (2 * pi / day))
  outputs['Year sin'] = tf.math.sin(timestamp_s * (2 * pi / year))
  outputs['Year cos'] = tf.math.cos(timestamp_s * (2 * pi / year))

  # Delete timestamp feature
  del outputs['Date Time']

  # Declare final list of features
  FINAL_FEATURE_LIST =  ["p (mbar)",
    "T (degC)",
    "rh (%)", 
    "VPmax (mbar)", 
    "VPdef (mbar)", 
    "sh (g/kg)",
    "rho (g/m**3)",
    "Wx",
    "Wy",
    "Day sin",
    'Day cos',
    'Year sin',
    'Year cos'
    ]

  # Scale all features
  for key in FINAL_FEATURE_LIST:
    outputs[key] = tft.scale_to_0_1(outputs[key])

  return outputs

def get_latest_ts(df):
   date_time_train_boundary = df.iloc[TRAIN_SPLIT - 1].name
   return date_time_train_boundary

def format_ts(date_time_train_boundary):
   date_time_train_boundary = datetime.strptime(date_time_train_boundary, '%d.%m.%Y %H:%M:%S')
   date_time_train_boundary = bytes(str(datetime.timestamp(date_time_train_boundary)), 'utf-8')
   return date_time_train_boundary

def partition_fn(line, num_partitions):
  '''
  Partition function to work with Beam.partition

  Args:
    line (string) - One record in the CSV file.
    num_partition (integer) - Number of partitions. Required argument by Beam. Unused in this function.

  Returns:
    0 or 1 (integer) - 0 if line timestamp is below the date time boundary, 1 otherwise. 
  '''

  # Split the CSV string to a list
  line_split = line.split(b',')

  # Get the timestamp of the current line
  line_dt = line_split[date_time_idx]

  # Check if it is above or below the date time boundary
  partition_num = int(line_dt > date_time_train_boundary)

  return partition_num

def read_and_transform_data(working_dir):
  '''
  Reads a CSV File and preprocesses the data using TF Transform

  Args:
    working_dir (string) - directory to place TF Transform outputs
  
  Returns:
    transform_fn - transformation graph
    transformed_train_data - transformed training examples
    transformed_test_data - transformed test examples
    transformed_metadata - transform output metadata
  '''

  # Delete TF Transform if it already exists
  if os.path.exists(working_dir):
    shutil.rmtree(working_dir)

  with beam.Pipeline() as pipeline:
      with tft_beam.Context(temp_dir=os.path.join(working_dir, TRANSFORM_TEMP_DIR)):
        read_input_fn = beam.io.ReadFromText(INPUT_FILE, coder=beam.coders.BytesCoder(), skip_header_lines=1)
        cleaner_fn = beam.Map(clean_fn)
        partition_data = beam.Partition(partition_fn, 2)
        transformed_data_coder = RecordBatchToExamplesEncoder(transformed_metadata.schema)
        write_path_train = os.path.join(working_dir, TRANSFORM_TRAIN_FILENAME)
        write_path_test = os.path.join(working_dir, TRANSFORM_TRAIN_FILENAME)
        encoder_fn = lambda batch, _: transformed_data_coder.encode(batch)
        write_path_fn = os.path.join(working_dir)
        transform_fn = tft_beam.AnalyzeAndTransformDataset(preprocessing_fn, output_record_batches=True)
        transform_fn_test = tft_beam.TransformDataset(output_record_batches=True)


        # Create a TFXIO to read the data with the schema. You need
        # to list all columns in order since the schema doesn't specify the
        # order of columns in the csv.
        csv_tfxio = tfxio.BeamRecordCsvTFXIO(
              physical_format='text',
              column_names=ordered_columns,
              schema=RAW_DATA_SCHEMA
        )
        read_data_with_schema = csv_tfxio.BeamSource()

        # Get the raw data metadata
        RAW_DATA_METADATA = csv_tfxio.TensorAdapterConfig()

        
        # Read the input CSV and clean the data
        raw_data = (
            pipeline
            | 'ReadTrainData' >> read_input_fn
            | 'CleanLines' >> cleaner_fn
        )

        # Partition the dataset into train and test sets using the partition_fn defined earlier.    
        raw_train_data, raw_test_data = (
            raw_data
            | 'TrainTestSplit' >> partition_data
        )

        # Parse the raw train data into inputs for TF Transform
        raw_train_data = (
            raw_train_data 
            | 'DecodeTrainData' >> read_data_with_schema
        )
        
        # Pair the train data with the metadata into a tuple
        raw_train_dataset = (raw_train_data, RAW_DATA_METADATA)

        # Training data transformation. The TFXIO (RecordBatch) output format
        # is chosen for improved performance.
        (transformed_train_data, transformed_metadata) , transform_fn = (
            raw_train_dataset 
            | transform_fn
        )

        # Parse the raw data into inputs for TF Transform
        raw_test_data = (
            raw_test_data
            | 'DecodeTestData' >> read_data_with_schema
        )
        
        # Pair the test data with the metadata into a tuple
        raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)
        
        # Now apply the same transform function to the test data.
        # You don't need the transformed data schema. It's the same as before.
        transformed_test_data, _ = (
            (raw_test_dataset, transform_fn) 
            | transform_fn_test
        )
        
        # Encode transformed train data and write to disk
        _ = (
            transformed_train_data
            | 'EncodeTrainData' >> beam.FlatMapTuple(encoder_fn)
            | 'WriteTrainData' >> beam.io.WriteToTFRecord(write_path_train)
            )

        # Encode transformed test data and write to disk
        _ = (
            transformed_test_data
            | 'EncodeTestData' >> beam.FlatMapTuple(encoder_fn)
            | 'WriteTestData' >> beam.io.WriteToTFRecord(write_path_test)
        )
        
        # Write transform function to disk
        _ = (
                transform_fn
                | 'WriteTransformFn' >>  tft_beam.WriteTransformFn(write_path_fn)
          )

         
  return transform_fn, transformed_train_data, transformed_test_data, transformed_metadata

def main():
  return read_and_transform_data(WORKING_DIR)

def get_output_transform_component(WORKING_DIR):
   tf_transform_output = tft.TFTransformOutput(os.path.join(WORKING_DIR))
   return tf_transform_output

def get_index_label_key(tf_transform_output, LABEL_KEY):
   index_of_label = list(tf_transform_output.transformed_feature_spec().keys()).index(LABEL_KEY)
   return index_of_label

def parse_function(example_proto):
    
    feature_spec = tf_transform_output.transformed_feature_spec()
    
    # Define features with the example_proto (transformed data) and the feature_spec using tf.io.parse_single_example 
    features = tf.io.parse_single_example(example_proto, feature_spec)
    values = list(features.values())
    values[index_of_label], values[len(features) - 1] = values[len(features) - 1], values[index_of_label]
    
    # Stack the values along the first axis
    stacked_features = tf.stack(values, axis=0)

    return stacked_features

def map_features_target(elements):
    features = elements[:HISTORY_SIZE]
    target = elements[-1:,-1]
    return (features, target)

def get_windowed_dataset(path):
        
    # Instantiate a tf.data.TFRecordDataset passing in the appropiate path
    dataset = tf.data.TFRecordDataset(path)
    
    # Use the dataset's map method to map the parse_function
    dataset = dataset.map(parse_function)
    
    # Use the window method with expected total size. Define stride and set drop_remainder to True
    dataset = dataset.window(HISTORY_SIZE + FUTURE_TARGET, shift=SHIFT, stride=OBSERVATIONS_PER_HOUR, drop_remainder=True)
    
    # Use the flat_map method passing in an anonymous function that given a window returns window.batch(HISTORY_SIZE + FUTURE_TARGET)
    dataset = dataset.flat_map(lambda window: window.batch(HISTORY_SIZE + FUTURE_TARGET))
    
    # Use the map method passing in the previously defined map_features_target function
    dataset = dataset.map(map_features_target) 
    
    # Use the batch method and pass in the appropiate batch size
    dataset = dataset.batch(BATCH_SIZE)

    return dataset

def get_tfrecord_files(WORKING_DIR, TRANSFORM_TRAIN_FILENAME, TRANSFORM_TEST_FILENAME):
    train_tfrecord_files = tf.io.gfile.glob(os.path.join(WORKING_DIR, TRANSFORM_TRAIN_FILENAME + '*'))
    test_tfrecord_files = tf.io.gfile.glob(os.path.join(WORKING_DIR, TRANSFORM_TEST_FILENAME + '*'))
    return train_tfrecord_files, test_tfrecord_files

def generate_windows(train_tfrecord_files, test_tfrecord_files):
    windowed_train_dataset = get_windowed_dataset(train_tfrecord_files[0])
    windowed_test_dataset = get_windowed_dataset(test_tfrecord_files[0])
    return windowed_train_dataset, windowed_test_dataset

def preview_examples(tf_transform_output, windowed_train_dataset):
    ordered_feature_spec_names = tf_transform_output.transformed_feature_spec().keys()

    # Preview an example in the train dataset
    for features, target  in windowed_train_dataset.take(1):
        print(f'Shape of input features for a batch: {features.shape}')
        print(f'Shape of targets for a batch: {target.shape}\n')

        print(f'INPUT FEATURES:')
        for value, name in zip(features[0][0].numpy(), ordered_feature_spec_names):
            print(f'{name} : {value}') 

        print(f'\nTARGET TEMPERATURE: {target[0][0]}')

EXECUTE = os.system
DATA_DIR = '/content/data/'
INPUT_FILE = os.path.join(DATA_DIR, 'jena_climate_2009_2016.csv')
TIMESTAMP_FEATURES = ["Date Time"]
NUMERIC_FEATURES = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)", 
    "rh (%)", 
    "VPmax (mbar)", 
    "VPact (mbar)", 
    "VPdef (mbar)", 
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]
FEATURES_TO_REMOVE = ["Tpot (K)", "Tdew (degC)","VPact (mbar)" , "H2OC (mmol/mol)", "max. wv (m/s)"]
TRAIN_SPLIT = 300000
RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in TIMESTAMP_FEATURES] +
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in NUMERIC_FEATURES]
)
RAW_DATA_SCHEMA = tft.tf_metadata.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC)
WORKING_DIR = 'transform_dir'
TRANSFORM_TRAIN_FILENAME = 'transform_train'
TRANSFORM_TEST_FILENAME = 'transform_test'
TRANSFORM_TEMP_DIR = 'tft_temp'
LABEL_KEY = 'T (degC)'
OBSERVATIONS_PER_HOUR = 6
HISTORY_SIZE = 120
FUTURE_TARGET = 12
BATCH_SIZE = 72
SHIFT = 1

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
ordered_columns = TIMESTAMP_FEATURES + NUMERIC_FEATURES
date_time_idx = ordered_columns.index(TIMESTAMP_FEATURES[0])
wv_idx = ordered_columns.index('wv (m/s)')

EXECUTE(f"wget -nc https://raw.githubusercontent.com/https-deeplearning-ai/MLEP-public/main/course2/week4-ungraded-lab/data/jena_climate_2009_2016.csv -P {DATA_DIR}")

# Code starts here

# Way #1: Using Pandas
df = read_input_data(INPUT_FILE)
visualize_plots(df, NUMERIC_FEATURES)
wv = set_velocity_outliers(df)
bad_max_wv = set_max_wind_vel(df)
visualize_plots(df, NUMERIC_FEATURES)
show_correlation_heatmap(df)
date_time_train_boundary = get_latest_ts(df)
date_time_train_boundary = format_ts(date_time_train_boundary)

# Way #2: Using tensorflow transform
transform_fn, transformed_train_data, trainsformed_test_data, transformed_metadata = main()
tf_transform_output = get_output_transform_component(WORKING_DIR)
index_of_label = get_index_label_key(tf_transform_output, LABEL_KEY)
train_tfrecord_files, test_tfrecord_files = get_tfrecord_files(WORKING_DIR, TRANSFORM_TRAIN_FILENAME, TRANSFORM_TEST_FILENAME)
windowed_train_dataset, windowed_test_dataset = generate_windows(train_tfrecord_files, test_tfrecord_files)
preview_examples(tf_transform_output, windowed_train_dataset)
preview_examples(tf_transform_output, windowed_test_dataset)
