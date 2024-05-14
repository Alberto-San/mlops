import tensorflow as tf
import tfx

# TFX components
from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform
from tfx.components import ImporterNode

# TFX libraries
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# For performing feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# For feature visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# Utilities
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf.json_format import MessageToDict
from  tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
import os
import pprint
import tempfile
import pandas as pd
from util import get_records

# Import mlmd and utilities
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
from util import display_types, display_artifacts, display_properties


def prepare_dataset(TRAINING_DATA):
    df = pd.read_csv(TRAINING_DATA)
    df_num = df.copy()
    cat_columns = ['Wilderness_Area', 'Soil_Type']
    label_column = ['Cover_Type']
    df_num.drop(cat_columns, axis=1, inplace=True)
    df_num.drop(label_column, axis=1, inplace=True)
    X = df_num.values
    y = df[label_column].values

    select_k_best = SelectKBest(score_func=f_classif, k=8)
    X_new = select_k_best.fit_transform(X,y)
    features_mask = select_k_best.get_support()
    reqd_cols = pd.DataFrame({'Columns': df_num.columns, 'Retain': features_mask})
    feature_names = list(df_num.columns[features_mask])
    feature_names = feature_names + cat_columns + label_column
    df_select = df[feature_names]
    df_select.to_csv(TRAINING_DATA_FSELECT, index=False)

def get_schema_uri(schema_gen):
    try:
        # Get the schema uri
        schema_uri = schema_gen.outputs['schema']._artifacts[0].uri
        
    # for grading since context.run() does not work outside the notebook
    except IndexError:
        print("context.run() was no-op")
        schema_path = './pipeline/SchemaGen/schema'
        dir_id = os.listdir(schema_path)[0]
        schema_uri = f'{schema_path}/{dir_id}'

    return schema_uri

def load_schema(schema_uri):
    schema = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))
    return schema

def modify_domains(schema):
    tfdv.set_domain(schema, 'Hillshade_9am', schema_pb2.IntDomain(name='Hillshade_9am', min=0, max=255))
    tfdv.set_domain(schema, 'Hillshade_Noon', schema_pb2.IntDomain(name='Hillshade_Noon', min=0, max=255))
    tfdv.set_domain(schema, 'Slope', schema_pb2.IntDomain(name='Slope', min=0, max=90))
    tfdv.set_domain(schema, 'Cover_Type', schema_pb2.IntDomain(name='Cover_Type', min=0, max=6, is_categorical=True))
    return schema

def create_serving_data(TRAINING_DATA, SERVING_DATA):
    serving_data = pd.read_csv(TRAINING_DATA, nrows=100)
    serving_data.drop(columns='Cover_Type', inplace=True)
    serving_data.to_csv(SERVING_DATA, index=False)
    del serving_data

def define_environment(schema, env):
    schema.default_environment.append(env)
    return schema

def drop_feature_environment(schema, env, feature):
    tfdv.get_feature(schema, feature).not_in_environment.append(env)
    return schema

def save_schema(schema, schema_file):
    tfdv.write_schema_text(schema, schema_file)

def generate_schema(TRAINING_DIR_FSELECT, context, display=False):
    # 1
    example_gen = CsvExampleGen(input_base=TRAINING_DIR_FSELECT)
    context.run(example_gen)

    # 2
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    context.run(statistics_gen)
    
    # 3
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    context.run(schema_gen)

    # 4
    schema_uri = get_schema_uri(schema_gen)
    schema = load_schema(schema_uri)
    schema = modify_domains(schema)

    if display:
        context.show(statistics_gen.outputs['statistics'])
        context.show(schema_gen.outputs['schema'])
        tfdv.display_schema(schema=schema) 

    return schema, example_gen, statistics_gen, schema_gen

def generate_statistics_from_csv(SERVING_DATA, schema):
    stats_options = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
    serving_stats = tfdv.generate_statistics_from_csv(SERVING_DATA, stats_options=stats_options)
    return serving_stats

def find_anomalies(serving_stats, schema):
    anomalies = tfdv.validate_statistics(serving_stats, schema=schema)
    tfdv.display_anomalies(anomalies)

def find_anomalies_using_component(statistics_gen, user_schema_importer):
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=user_schema_importer.outputs['result']
    )
    context.run(example_validator)
    context.show(statistics_gen.outputs['statistics']) 


def load_schema(schema_file):
    return tfdv.load_schema_text(schema_file)

def getting_schema_from_file(UPDATED_SCHEMA_DIR):
    user_schema_importer = ImporterNode(
        instance_name='import_user_schema',
        source_uri=UPDATED_SCHEMA_DIR,
        artifact_type=standard_artifacts.Schema
    )
    context.run(user_schema_importer, enable_cache=False)
    return user_schema_importer

def process_schema(schema):
    schema = define_environment(schema, env='TRAINING')
    schema = define_environment(schema, env='SERVING')
    schema = drop_feature_environment(schema, env='SERVING', feature='Cover_Type')
    return schema

def print_statistics(example_gen, user_schema_importer):
    statistics_gen_updated = StatisticsGen(
        examples=example_gen.outputs['examples'], 
        stats_options=tfdv.StatsOptions(infer_type_from_schema=True),
        schema=user_schema_importer.outputs['result']
    )
    context.run(statistics_gen_updated)
    context.show(statistics_gen_updated.outputs['statistics'])
    return 

def run_transformation(example_gen, schema_gen, _cover_transform_module_file):
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(_cover_transform_module_file)
    )
    context.run(transform, enable_cache=False)
    return transform

def get_transform_uri(transform):
    try:
        transform_uri = transform.outputs['transformed_examples'].get()[0].uri

    # for grading since context.run() does not work outside the notebook
    except IndexError:
        print("context.run() was no-op")
        examples_path = './pipeline/Transform/transformed_examples'
        dir_id = os.listdir(examples_path)[0]
        transform_uri = f'{examples_path}/{dir_id}'
    return transform_uri

def print_records(transform_uri):
    pp = pprint.PrettyPrinter()
    train_uri = os.path.join(transform_uri, 'train')
    tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]
    transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
    sample_records_xf = get_records(transformed_dataset, 3)
    pp.pprint(sample_records_xf)

def get_parent_artifacts(store, artifact):    
    # Get the artifact id of the input artifact
    artifact_id = artifact.id
    
    # Get events associated with the artifact id
    artifact_id_events = store.get_events_by_artifact_ids([artifact_id])
    
    # From the `artifact_id_events`, get the execution ids of OUTPUT events.
    # Cast to a set to remove duplicates if any.
    execution_id = set( 
        event.execution_id
        for event in artifact_id_events # @REPLACE
        if event.type == metadata_store_pb2.Event.OUTPUT # @REPLACE
    )
    
    # Get the events associated with the execution_id
    execution_id_events = store.get_events_by_execution_ids(execution_id)

    # From execution_id_events, get the artifact ids of INPUT events.
    # Cast to a set to remove duplicates if any.
    parent_artifact_ids = set( 
        event.artifact_id
        for event in execution_id_events
        if event.type == metadata_store_pb2.Event.INPUT
    )
    
    # Get the list of artifacts associated with the parent_artifact_ids
    parent_artifact_list = [artifact for artifact in store.get_artifacts_by_id(parent_artifact_ids)]

    
    return parent_artifact_list


execute = os.system
DATA_DIR = './data'
TRAINING_DIR = f'{DATA_DIR}/training'
TRAINING_DATA = f'{TRAINING_DIR}/dataset.csv'
TRAINING_DIR_FSELECT = f'{TRAINING_DIR}/fselect'
TRAINING_DATA_FSELECT = f'{TRAINING_DIR_FSELECT}/dataset.csv'
PIPELINE_DIR = './pipeline'
SERVING_DIR = f'{DATA_DIR}/serving'
SERVING_DATA = f'{SERVING_DIR}/serving_dataset.csv'
UPDATED_SCHEMA_DIR = f'{PIPELINE_DIR}/updated_schema'
schema_file = os.path.join(UPDATED_SCHEMA_DIR, 'schema.pbtxt')
_cover_transform_module_file = 'cover_transform.py'
transform_script = """
import tensorflow as tf
import tensorflow_transform as tft

import cover_constants

_SCALE_MINMAX_FEATURE_KEYS = cover_constants.SCALE_MINMAX_FEATURE_KEYS
_SCALE_01_FEATURE_KEYS = cover_constants.SCALE_01_FEATURE_KEYS
_SCALE_Z_FEATURE_KEYS = cover_constants.SCALE_Z_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = cover_constants.VOCAB_FEATURE_KEYS
_HASH_STRING_FEATURE_KEYS = cover_constants.HASH_STRING_FEATURE_KEYS
_LABEL_KEY = cover_constants.LABEL_KEY
_transformed_name = cover_constants.transformed_name

def preprocessing_fn(inputs):

    features_dict = {}

    ### START CODE HERE ###
    for feature in _SCALE_MINMAX_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using scaling of min_max function
        # Hint: Use tft.scale_by_min_max by passing in the respective column
        features_dict[_transformed_name(feature)] = tft.scale_by_min_max(data_col)

    for feature in _SCALE_01_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using scaling of 0 to 1 function
        # Hint: tft.scale_to_0_1
        features_dict[_transformed_name(feature)] = tft.scale_to_0_1(data_col)

    for feature in _SCALE_Z_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using scaling to z score
        # Hint: tft.scale_to_z_score
        features_dict[_transformed_name(feature)] = tft.scale_to_z_score(data_col)

    for feature in _VOCAB_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform using vocabulary available in column
        # Hint: Use tft.compute_and_apply_vocabulary
        features_dict[_transformed_name(feature)] = tft.compute_and_apply_vocabulary(data_col)

    for feature in _HASH_STRING_FEATURE_KEYS:
        data_col = inputs[feature] 
        # Transform by hashing strings into buckets
        # Hint: Use tft.hash_strings with the param hash_buckets set to 10
        features_dict[_transformed_name(feature)] = tft.hash_strings(data_col, hash_buckets=10)
    
    ### END CODE HERE ###  

    # No change in the label
    features_dict[_LABEL_KEY] = inputs[_LABEL_KEY]

    return features_dict
"""

execute(f"mkdir -p {TRAINING_DIR}")
execute(f"wget -nc https://storage.googleapis.com/workshop-datasets/covertype/full/dataset.csv -P {TRAINING_DIR}")
execute(f"mkdir -p {TRAINING_DIR_FSELECT}")
execute(f"mkdir -p {SERVING_DIR}")
execute(f"mkdir -p {UPDATED_SCHEMA_DIR}")
execute(f"CAT << EOF > {_cover_transform_module_file}\n {transform_script} EOF")

prepare_dataset(TRAINING_DATA)
create_serving_data(TRAINING_DATA, SERVING_DATA)

context = InteractiveContext(pipeline_root=PIPELINE_DIR)
schema, example_gen, statistics_gen, schema_gen = generate_schema(TRAINING_DIR_FSELECT, context, display=True)
serving_stats = generate_statistics_from_csv(SERVING_DATA, schema)
find_anomalies(serving_stats, schema)

# Curaing schema
schema = process_schema(schema)
find_anomalies(serving_stats, schema)

# Update schema and save it into local
save_schema(schema, schema_file)

# Using new schema
new_schema = load_schema(schema_file)
tfdv.display_schema(schema=new_schema)

# Update schema
user_schema_importer = getting_schema_from_file(UPDATED_SCHEMA_DIR)
context.show(user_schema_importer.outputs['result'])

print_statistics(example_gen, user_schema_importer)
find_anomalies_using_component(statistics_gen, user_schema_importer)

# Feature Eng
transform = run_transformation(example_gen, schema_gen, _cover_transform_module_file)
transform_uri = get_transform_uri(transform)
print_records(os.path.join(transform_uri, 'train'))

# Interacting with MLMD
connection_config = context.metadata_connection_config
store = mlmd.MetadataStore(connection_config)
base_dir = connection_config.sqlite.filename_uri.split('metadata.sqlite')[0]

types = store.get_artifact_types()
df_artifacts = display_types(types)
print(df_artifacts.head(10))

schema_list = store.get_artifacts_by_type('Schema')
df_artifacts = display_artifacts(store, schema_list, base_dir)
print(df_artifacts.head(10))

statistics_artifacts = store.get_artifacts_by_type('ExampleStatistics')
statistics_artifact = statistics_artifacts[-1]
print(display_properties(store, statistics_artifact).head(10))

# Get parent of an artifact: what was the inputs that was used to generate that artifact
artifact_instances = store.get_artifacts_by_type('TransformGraph')
artifact_instance = artifact_instances[0]
parent_artifacts = get_parent_artifacts(store, artifact_instance)
print(display_artifacts(store, parent_artifacts, base_dir).head(10))

