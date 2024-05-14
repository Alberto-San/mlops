# Import mlmd and utilities
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2
import tensorflow as tf
import tensorflow_data_validation as tfdv

from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import ImporterNode
from tfx.types import standard_artifacts

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict
from tensorflow_metadata.proto.v0 import schema_pb2

import os
import pprint

def run_component(component, context):
    context.run(example_gen)
    return component

def load_schema(schema_gen):
    schema_uri = schema_gen.outputs['schema']._artifacts[0].uri
    schema = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))
    return schema

def modified_age_domain(schema):
    tfdv.set_domain(schema, 'age', schema_pb2.IntDomain(name='age', min=17, max=90))
    return schema

def create_schema_environment(schema, env):
    schema.default_environment.append(env)
    return schema

def omit_label(schema, label, env):
    tfdv.get_feature(schema, label).not_in_environment.append(env)
    return schema

def create_dir(_updated_schema_dir):
    os.system(f"mkdir -p {_updated_schema_dir}")


def generate_schema_artifact(context):
    user_schema_importer = ImporterNode(
        instance_name='import_user_schema',
        source_uri=_updated_schema_dir,
        artifact_type=standard_artifacts.Schema
    )
    context.run(user_schema_importer, enable_cache=False)
    return user_schema_importer

def print_artifacts_metastore(artifact_types):
    return print(f"artifact names are: {' ,'.join([artifact_type.name for artifact_type in artifact_types])}")

def print_all_schemas(schema_list):
    print([(f'schema uri: {schema.uri}', f'schema id:{schema.id}') for schema in schema_list])



pp = pprint.PrettyPrinter()
_pipeline_root = './pipeline/'
_data_root = './data/census_data'
_data_filepath = os.path.join(_data_root, 'adult.data')
_updated_schema_dir = f'{_pipeline_root}/updated_schema'
schema_file = os.path.join(_updated_schema_dir, 'schema.pbtxt')

create_dir(_updated_schema_dir)
context = InteractiveContext(pipeline_root=_pipeline_root)
example_gen = run_component(CsvExampleGen(input_base=_data_root), context)
statistics_gen = run_component(StatisticsGen(examples=example_gen.outputs['examples']), context)
schema_gen = run_component(SchemaGen(statistics=statistics_gen.outputs['statistics']), context)

user_schema = load_schema(schema_gen)\
.map(modified_age_domain)\ 
.map(lambda schema: create_schema_environment(schema, "TRAINING"))\
.map(lambda schema: create_schema_environment(schema, "SERVING"))\
.map(lambda schema: omit_label(schema, label="label", env="SERVING"))\
.map(lambda schema_: tfdv.write_schema_text(schema_, schema_file))\
.flatMap(lambda : generate_schema_artifact(context))

example_validator = run_component(
    ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=user_schema.outputs['result']
    )
)
context.show(example_validator.outputs['anomalies'])

################# Post #####################################
context = ...
connection_config = context.metadata_connection_config
store = mlmd.MetadataStore(connection_config)
artifact_types = store.get_artifact_types()
print_artifacts_metastore(artifact_types)
schema_list = store.get_artifacts_by_type('Schema')
print_all_schemas(schema_list)
examples_anomalies = store.get_artifacts_by_type('ExampleAnomalies')
sample_anomalies = examples_anomalies[0]
anomalies_id_events = store.get_events_by_artifact_ids([example_anomalies.id]) # output
anomalies_id_event = anomalies_id_events[0]
anomalies_execution_id = anomalies_id_event.execution_id
events_executions = store.get_events_by_execution_ids([anomalies_execution_id]) # outputs and inputs
inputs_to_exval = [event.artifact_id for event in events_execution 
                       if event.type == metadata_store_pb2.Event.INPUT]





