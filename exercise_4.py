import os
import pprint
import tempfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import absl
import tensorflow as tf
tf.get_logger().propagate = False
pp = pprint.PrettyPrinter()

import tfx
from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.types import Channel
from tfx.utils.dsl_utils import external_input
from tfx.components.transform.component import Transform
import tensorflow_transform as tft

from google.protobuf.json_format import MessageToDict

Monad = ...

def show_correlation_heatmap(dataframe):
    plt.figure(figsize=(20,20))
    cor = dataframe.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
    plt.show()

def generate_base_data(_data_filepath):
    df = pd.read_csv(f'{_data_filepath}')
    df.columns=df.columns.str.replace('"','')
    df.to_csv(f'{_data_filepath}', index=False)
    return df

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

def get_records(dataset, num_records):
    '''Extracts records from the given dataset.
    Args:
        dataset (TFRecordDataset): dataset saved by ExampleGen
        num_records (int): number of records to preview
    '''
    
    # initialize an empty list
    records = []
    
    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):
        
        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()
        
        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()
        
        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)
        
        # convert the protocol bufffer message to a Python dictionary
        example_dict = (MessageToDict(example))
        
        # append to the records list
        records.append(example_dict)
        
    return records

def parse_function(example_proto, tf_transform_output, index_of_label):
    
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

def get_dataset_windowed(path, tf_transform_output, index_of_label):
        
    # Instantiate a tf.data.TFRecordDataset passing in the appropiate path
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP')
    
    # Use the dataset's map method to map the parse_function
    dataset = dataset.map(lambda ds: parse_function(ds, tf_transform_output, index_of_label))
    
    # Use the window method with expected total size. Define stride and set drop_remainder to True
    dataset = dataset.window(HISTORY_SIZE + FUTURE_TARGET, shift=SHIFT, stride=OBSERVATIONS_PER_HOUR, drop_remainder=True)
    
    # Use the flat_map method passing in an anonymous function that given a window returns window.batch(HISTORY_SIZE + FUTURE_TARGET)
    dataset = dataset.flat_map(lambda window: window.batch(HISTORY_SIZE + FUTURE_TARGET))
    
    # Use the map method passing in the previously defined map_features_target function
    dataset = dataset.map(map_features_target) 
    
    # Use the batch method and pass in the appropiate batch size
    dataset = dataset.batch(BATCH_SIZE)

    return dataset

def prepare_env():
    Monad([
        "mkdir pipeline",
        "mkdir -p data/climate",
        f"wget -nc https://raw.githubusercontent.com/https-deeplearning-ai/MLEP-public/main/course2/week4-ungraded-lab/data/jena_climate_2009_2016.csv -P {_data_root}",
        _weather_constants_module_module,
        _weather_transform_module_module
    ]).map(os.system)

def generate_context(_pipeline_root):
    context = InteractiveContext(pipeline_root=_pipeline_root)
    return context

def input_generator(_data_root, context):
    example_gen = CsvExampleGen(input_base=_data_root)
    context.run(example_gen)

def generate_statistics(example_gen):
  statistics_gen = StatisticsGen(
      examples=example_gen.outputs['examples'])
  context.run(statistics_gen)
  context.show(statistics_gen.outputs['statistics'])
  return statistics_gen

def generate_schema(statistics_gen):
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], 
      infer_feature_shape=True
      )
  context.run(schema_gen)
  context.show(schema_gen.outputs['schema'])
  return schema_gen

def get_anomalus_data(statistics_gen, schema_gen):
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  context.run(example_validator)
  context.show(example_validator.outputs['anomalies'])
  return example_validator

def apply_transformations(example_gen, schema_gen, module_file):
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)
  context.run(transform)
  return transform

def print_transform_records(transform):
  train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'train')
  tfrecord_filenames = [os.path.join(train_uri, name)
                        for name in os.listdir(train_uri)]
  transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
  sample_records_xf = get_records(transformed_dataset, 3)
  pp.pprint(sample_records_xf)

def get_label_index_dataset(tf_transform_output, LABEL_KEY):
  index_of_label = list(tf_transform_output.transformed_feature_spec().keys()).index(LABEL_KEY)
  return index_of_label

def get_uri_transformed_examples(transform):
  working_dir = transform.outputs['transformed_examples'].get()[0].uri
  return working_dir

def filename_compress_train_examples(working_dir):
  train_tfrecord_files = os.listdir(working_dir + '/train')[0]
  return train_tfrecord_files

_pipeline_root = './pipeline/'
_data_root = './data/climate'
_data_filepath = os.path.join(_data_root, 'jena_climate_2009_2016.csv')
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
feature_keys = [
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
_weather_constants_module_file = 'weather_constants.py'
_weather_constants_module_module = f""" CAT << EOF > {_weather_constants_module_file}
SELECTED_NUMERIC_FEATURES = ['T (degC)', 'VPmax (mbar)', 'VPdef (mbar)', 'sh (g/kg)','rho (g/m**3)']

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
EOF
"""
_weather_transform_module_file = 'weather_transform.py'
_weather_transform_module_module = f""" CAT << EOF > {_weather_transform_module_file}
import tensorflow as tf
import tensorflow_transform as tft

import weather_constants
import tensorflow_addons as tfa

import math as m

# Features to filter out
FEATURES_TO_REMOVE = ["Tpot (K)", "Tdew (degC)","VPact (mbar)" , "H2OC (mmol/mol)", "max. wv (m/s)"]

# Unpack the contents of the constants module
_SELECTED_NUMERIC_FEATURE_KEYS = weather_constants.SELECTED_NUMERIC_FEATURES
_transformed_name = weather_constants.transformed_name

# Define the transformations
def preprocessing_fn(inputs):
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

    # Delete `wv (m/s)` after getting the wind vector
    del outputs['wv (m/s)']

    # Get day and year in seconds
    day = tf.cast(24*60*60, tf.float32)
    year = tf.cast((365.2425)*day, tf.float32)

    # Convert `Date Time` column into timestamps in seconds (using tfa helper function)
    timestamp_s = tfa.text.parse_time(outputs['Date Time'], time_format='%d.%m.%Y %H:%M:%S', output_unit='SECOND')
    timestamp_s = tf.cast(timestamp_s, tf.float32)
    
    # Convert timestamps into periodic signals
    outputs['Day sin'] = tf.math.sin(timestamp_s * (2 * pi / day))
    outputs['Day cos'] = tf.math.cos(timestamp_s * (2 * pi / day))
    outputs['Year sin'] = tf.math.sin(timestamp_s * (2 * pi / year))
    outputs['Year cos'] = tf.math.cos(timestamp_s * (2 * pi / year))

    # Delete unneeded columns
    del outputs['Date Time']
    del outputs['wd (deg)']

    # Final feature list
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

    # Scale selected numeric features
    for key in _SELECTED_NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_0_1(outputs[key])

    return outputs
EOF
"""
LABEL_KEY = 'T (degC)'
OBSERVATIONS_PER_HOUR = 6
HISTORY_SIZE = 120
FUTURE_TARGET = 12
BATCH_SIZE = 72
SHIFT = 1

_ = prepare_env()
df = generate_base_data(_data_filepath)
visualize_plots(df, feature_keys)
show_correlation_heatmap(df)

context = generate_context(_pipeline_root)
example_gen = input_generator(_data_root, context)
statistics_gen = generate_statistics(example_gen)
schema_gen = generate_schema(statistics_gen)
_ = get_anomalus_data(statistics_gen, schema_gen)
transform = apply_transformations(example_gen, schema_gen, module_file=os.path.abspath(_weather_transform_module_file))
print_transform_records(transform)

WORKING_DIR = transform.outputs['transform_graph'].get()[0].uri
tf_transform_output = tft.TFTransformOutput(os.path.join(WORKING_DIR))
index_of_label = get_label_index_dataset(tf_transform_output, LABEL_KEY)
working_dir = get_uri_transformed_examples(transform)
train_tfrecord_files = filename_compress_train_examples(working_dir)
train_dataset = get_dataset_windowed(
   path_to_train_tfrecord_files=os.path.join(working_dir, 'train', train_tfrecord_files), 
   tf_transform_output=tf_transform_output, 
   index_of_label=index_of_label
)

for features, target in train_dataset.take(1):
    print(f'Shape of input features for a batch: {features.shape}')
    print(f'Shape of targets for a batch: {target.shape}')
