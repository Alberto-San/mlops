from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2
import tensorflow as tf
import tensorflow_data_validation as tfdv
import urllib
import zipfile

def download_dataset():
    # Download the zip file from GCP and unzip it
    url = 'https://storage.googleapis.com/artifacts.tfx-oss-public.appspot.com/datasets/chicago_data.zip'
    zip, _ = urllib.request.urlretrieve(url)
    zipfile.ZipFile(zip).extractall()
    zipfile.ZipFile(zip).close()

def get_database(type):
    connection_config = metadata_store_pb2.ConnectionConfig()

    if type == "fake":
        connection_config.fake_database.SetInParent() 
    elif type == "sqlite":
        ...
    elif type == "blob":
        ...

    return connection_config

def create_dataset_artifact():
    data_artifact_type = metadata_store_pb2.ArtifactType()
    data_artifact_type.name = 'DataSet'
    data_artifact_type.properties['name'] = metadata_store_pb2.STRING
    data_artifact_type.properties['split'] = metadata_store_pb2.STRING
    data_artifact_type.properties['version'] = metadata_store_pb2.INT
    return data_artifact_type

def register_artifact(store, object, type):
    type = type.lower()
    if type == "artifact_type" or type == "schema_type":
        artifact_type_id = store.put_artifact_type(object)
        return artifact_type_id
    elif type == "execution_type":
        dv_execution_type_id = store.put_execution_type(object)
        return dv_execution_type_id
    elif type == "artifact":
        data_artifact_id = store.put_artifacts(object)
        return data_artifact_id
    elif type == "execution":
        dv_execution_id = store.put_executions(object)
        return dv_execution_id
    elif type == "event":
        id = store.put_events(object)
        return id
    elif type == "context_type":
        expt_context_type_id = store.put_context_type(object)
        return expt_context_type_id
    elif type == "context":
        expt_context_id = store.put_contexts(object)
        return expt_context_id
    elif type == "attribution and association":
        id = store.put_attributions_and_associations(object["expt_attribution"], object["expt_association"])
    else:
        raise(f"type {type.upper()} not supported")

def create_schema_artifact():
    schema_artifact_type = metadata_store_pb2.ArtifactType()
    schema_artifact_type.name = 'Schema'
    schema_artifact_type.properties['name'] = metadata_store_pb2.STRING
    schema_artifact_type.properties['version'] = metadata_store_pb2.INT
    return schema_artifact_type

def create_exec_type():
    dv_execution_type = metadata_store_pb2.ExecutionType()
    dv_execution_type.name = 'Data Validation'
    dv_execution_type.properties['state'] = metadata_store_pb2.STRING
    return dv_execution_type

def put_input_artifact_unit(data_artifact_type_id):
    data_artifact = metadata_store_pb2.Artifact()
    data_artifact.uri = './data/train/data.csv'
    data_artifact.type_id = data_artifact_type_id
    data_artifact.properties['name'].string_value = 'Chicago Taxi dataset'
    data_artifact.properties['split'].string_value = 'train'
    data_artifact.properties['version'].int_value = 1
    return data_artifact

def put_execution_unit(dv_execution_type_id):
    dv_execution = metadata_store_pb2.Execution()   
    dv_execution.type_id = dv_execution_type_id
    dv_execution.properties['state'].string_value = 'RUNNING'
    return dv_execution

def generate_input_event(data_artifact_id, dv_execution_id):
    # types: https://github.com/google/ml-metadata/blob/master/ml_metadata/proto/metadata_store.proto#L187
    input_event = metadata_store_pb2.Event()
    input_event.artifact_id = data_artifact_id
    input_event.execution_id = dv_execution_id
    input_event.type = metadata_store_pb2.Event.DECLARED_INPUT
    return input_event

def generate_schema():
    train_data = './data/train/data.csv'
    train_stats = tfdv.generate_statistics_from_csv(data_location=train_data)
    schema = tfdv.infer_schema(statistics=train_stats)
    tfdv.write_schema_text(schema, schema_file)

def generate_output_artifact_unit(schema_artifact_type_id):
    schema_artifact = metadata_store_pb2.Artifact()
    schema_artifact.uri = schema_file
    schema_artifact.type_id = schema_artifact_type_id
    schema_artifact.properties['version'].int_value = 1
    schema_artifact.properties['name'].string_value = 'Chicago Taxi Schema'
    return schema_artifact

def generate_output_event(schema_artifact_id, dv_execution_id):
    output_event = metadata_store_pb2.Event()
    output_event.artifact_id = schema_artifact_id
    output_event.execution_id = dv_execution_id
    output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT
    return output_event

def update_exec_unit(dv_execution, dv_execution_id):
    dv_execution.id = dv_execution_id
    dv_execution.properties['state'].string_value = 'COMPLETED'
    return dv_execution

def create_context_type():
    expt_context_type = metadata_store_pb2.ContextType()
    expt_context_type.name = 'Experiment'
    expt_context_type.properties['note'] = metadata_store_pb2.STRING
    return expt_context_type

def generate_context(expt_context_type_id):
    expt_context = metadata_store_pb2.Context()
    expt_context.type_id = expt_context_type_id
    expt_context.name = 'Demo'
    expt_context.properties['note'].string_value = 'Walkthrough of metadata'
    return expt_context

def generate_attributions(schema_artifact_id, expt_context_id):
    expt_attribution = metadata_store_pb2.Attribution()
    expt_attribution.artifact_id = schema_artifact_id
    expt_attribution.context_id = expt_context_id
    return expt_attribution

def generate_associations(dv_execution_id, expt_context_id):
    expt_association = metadata_store_pb2.Association()
    expt_association.execution_id = dv_execution_id
    expt_association.context_id = expt_context_id
    return expt_association

schema_file = './schema.pbtxt'
connection_config = get_database(type="fake") 
store = metadata_store.MetadataStore(connection_config)
add_artifact = lambda object, type: register_artifact(store, object, type=type)

# generate placeholders
data_artifact_type = create_dataset_artifact()
data_artifact_type_id = add_artifact(data_artifact_type, type="artifact_type")
schema_artifact_type = create_schema_artifact()
schema_artifact_type_id = add_artifact(schema_artifact_type, type="schema_type")
dv_execution_type = create_exec_type()
dv_execution_type_id = add_artifact(dv_execution_type, type="execution_type")

# Put information
data_artifact = put_input_artifact_unit(data_artifact_type_id)
data_artifact_id = add_artifact([data_artifact], type="artifact")
data_artifact_id = data_artifact_id[0]

dv_execution = put_execution_unit(dv_execution_type_id)
dv_execution_id = add_artifact([dv_execution], type="execution")
dv_execution_id = dv_execution_id[0]

input_event = generate_input_event(data_artifact_id, dv_execution_id)
input_event_id = add_artifact([input_event], type="event")

generate_schema()
schema_artifact = generate_output_artifact_unit(schema_artifact_type_id)
schema_artifact_id = add_artifact([schema_artifact], type="artifact")
schema_artifact_id = schema_artifact_id[0]

output_event = generate_output_event(schema_artifact_id, dv_execution_id)
_ = add_artifact([output_event], type="event")

dv_execution = update_exec_unit(dv_execution, dv_execution_id)
_ = add_artifact([dv_execution], type="execution")

expt_context_type = create_context_type()
expt_context_type_id = add_artifact(expt_context_type, type="context_type")

expt_context = generate_context(expt_context_type_id)
expt_context_id = add_artifact([expt_context], type="context")
expt_context_id = expt_context_id[0]

expt_attribution = generate_attributions(schema_artifact_id, expt_context_id)
expt_association = generate_associations(dv_execution_id, expt_context_id)
add_artifact(
    {
        "expt_attribution": [expt_attribution],
        "expt_association": [expt_association]
    }, 
    type="attribution and association"
)


Monad = ...


row = 0
artifact_name = 'Schema'

input_event = Monad(artifact_name) \
.map(lambda artifact_name: store.get_artifacts_by_type(artifact_name)) \
.map(lambda schema_record: store.get_events_by_artifact_ids([schema_record[row].id])) \
.map(lambda event: store.get_events_by_execution_ids([event[row].execution_id])).value

print(input_event)
# [artifact_id: 1
# execution_id: 1
# type: DECLARED_INPUT
# milliseconds_since_epoch: 1715654486074
# , artifact_id: 2
# execution_id: 1
# type: DECLARED_OUTPUT
# milliseconds_since_epoch: 1715654488980
# ]

row_input = 0
Monad(input_event).map(lambda execution: store.get_artifacts_by_id([execution[row_input].artifact_id]))







