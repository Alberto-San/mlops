import tensorflow as tf
from tfx.components import CsvExampleGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Transform
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict
import os
import pprint

def generate_preprocessing_files():
    traffic_constants_cmd = ''' cat << EOF > traffic_constants.py
# Features to be scaled to the z-score
DENSE_FLOAT_FEATURE_KEYS = ['temp', 'snow_1h']

# Features to bucketize
BUCKET_FEATURE_KEYS = ['rain_1h']

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = {'rain_1h': 3}

# Feature to scale from 0 to 1
RANGE_FEATURE_KEYS = ['clouds_all']

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

# Features with string data types that will be converted to indices
VOCAB_FEATURE_KEYS = [
    'holiday',
    'weather_main',
    'weather_description'
]

# Features with int data type that will be kept as is
CATEGORICAL_FEATURE_KEYS = [
    'hour', 'day', 'day_of_week', 'month'
]

# Feature to predict
VOLUME_KEY = 'traffic_volume'

def transformed_name(key):
    return key + '_xf'
    EOF
    '''
    traffic_transform_cmd = ''' cat << EOF > traffic_transform.py
import tensorflow as tf
import tensorflow_transform as tft

import traffic_constants

# Unpack the contents of the constants module
_DENSE_FLOAT_FEATURE_KEYS = traffic_constants.DENSE_FLOAT_FEATURE_KEYS
_RANGE_FEATURE_KEYS = traffic_constants.RANGE_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = traffic_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = traffic_constants.VOCAB_SIZE
_OOV_SIZE = traffic_constants.OOV_SIZE
_CATEGORICAL_FEATURE_KEYS = traffic_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = traffic_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = traffic_constants.FEATURE_BUCKET_COUNT
_VOLUME_KEY = traffic_constants.VOLUME_KEY
_transformed_name = traffic_constants.transformed_name


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
    inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}

    ### START CODE HERE
    
    # Scale these features to the z-score.
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        # Scale these features to the z-score.
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])
            

    # Scale these feature/s from 0 to 1
    for key in _RANGE_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])
            

    # Transform the strings into indices 
    # hint: use the VOCAB_SIZE and OOV_SIZE to define the top_k and num_oov parameters
    for key in _VOCAB_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            inputs[key], 
            top_k=_VOCAB_SIZE, 
            num_oov_buckets=_OOV_SIZE
        )
            
            
            

    # Bucketize the feature
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], 
            _FEATURE_BUCKET_COUNT[key],
            always_return_num_quantiles=False
        )
            
            

    # Keep as is. No tft function needed.
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = inputs[key]

        
    # Use `tf.cast` to cast the label key to float32 and fill in the missing values.
    traffic_volume = tf.cast(_fill_in_missing(inputs[_VOLUME_KEY]), tf.float32)
  
    
    # Create a feature that shows if the traffic volume is greater than the mean and cast to an int
    outputs[_transformed_name(_VOLUME_KEY)] = tf.cast(  
        # Use `tf.greater` to check if the traffic volume in a row is greater than the mean of the entire traffic volumn column
        tf.greater(
            traffic_volume, 
            tft.mean(
                tf.cast(
                    traffic_volume, 
                    tf.float32
                )
            )
        ),
        tf.int64
    )                                        

    ### END CODE HERE
    return outputs


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor and convert to a dense tensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
        x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
          in the second dimension.
    Returns:
        A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(
                x.indices, 
                x.values, 
                [x.dense_shape[0], 1]
            ),
            default_value
        ),
        axis=1
    )
    '''
    

def get_artifact_uri(example_gen):
    # You can also check that programmatically with the following snippet. You can focus on the try block. The except and else block is needed mainly for grading. context.run() yields no operation when executed in a non-interactive environment (such as the grader script that runs outside of this notebook). In such scenarios, the URI must be manually set to avoid errors.
    try:
        # get the artifact object
        artifact = example_gen.outputs['examples'].get()[0]
    # for grading since context.run() does not work outside the notebook
    except IndexError:
        print("context.run() was no-op")
        examples_path = './metro_traffic_pipeline/CsvExampleGen/examples'
        dir_id = os.listdir(examples_path)[0]
        artifact_uri = f'{examples_path}/{dir_id}'
    else:
        artifact_uri = artifact.uri

    return artifact_uri

def get_records(dataset, num_records):
    '''Extracts records from the given dataset.
    Args:
        dataset (TFRecordDataset): dataset saved by ExampleGen
        num_records (int): number of records to preview
    '''
    
    # initialize an empty list
    records = []

    ### START CODE HERE
    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):
        
        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()
        
        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()
        
        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)
        
        # convert the protocol bufffer message to a Python dictionary
        example_dict = MessageToDict(example)
        
        # append to the records list
        records.append(example_dict)
        
    ### END CODE HERE
    return records


def print_records(component, set='train', N=3, compression_type="GZIP"):
    pp = pprint.PrettyPrinter()
    artifact_uri =  get_artifact_uri(component)
    train_uri = os.path.join(artifact_uri, set)
    tfrecord_filenames = [os.path.join(train_uri, name)
                        for name in os.listdir(train_uri)]
    dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type=compression_type)
    sample_records = get_records(dataset, N)
    pp.pprint(sample_records)



# location of the pipeline metadata store
generate_preprocessing_files()
_pipeline_root = 'metro_traffic_pipeline/'
context = InteractiveContext(pipeline_root=_pipeline_root)

_data_root = 'metro_traffic_pipeline/data'
example_gen = CsvExampleGen(input_base=_data_root)
context.run(example_gen)
print_records(example_gen, set='train', N=3, compression_type="GZIP")

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
context.run(statistics_gen)
context.show(statistics_gen.outputs['statistics'])

schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
context.run(schema_gen)
context.show(schema_gen.outputs['schema'])

example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'],schema=schema_gen.outputs['schema'])
context.run(example_validator)
context.show(example_validator.outputs['anomalies'])

transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(_traffic_transform_module_file)
    )
context.run(transform)

## Jupyter: https://github.com/amanchadha/coursera-machine-learning-engineering-for-prod-mlops-specialization/blob/main/C2%20-%20Machine%20Learning%20Data%20Lifecycle%20in%20Production/Week%202/C2W2_Assignment.ipynb
