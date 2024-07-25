

# Features with string data types that will be converted to indices
CATEGORICAL_FEATURE_KEYS = [
    'ocean_proximity'
]

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['longitude','latitude','total_rooms','total_bedrooms','population','households','median_income','median_house_value']

# Feature that can be grouped into buckets
BUCKET_FEATURE_KEYS = ['housing_median_age']

# Number of buckets used by tf.transform for encoding each bucket feature.
FEATURE_BUCKET_COUNT = {'housing_median_age':6}

# Feature that the model will predict
LABEL_KEY = 'ocean_proximity'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'


################
# Transform code
################


import tensorflow as tf
import tensorflow_transform as tft

import housing_constant

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = housing_constant.NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = housing_constant.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = housing_constant.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = housing_constant.FEATURE_BUCKET_COUNT
_LABEL_KEY = housing_constant.LABEL_KEY
_transformed_name = housing_constant.transformed_name


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1( # appending transformed features into empy dictionary
            inputs[key])
    
    # Bucketize these features
    for key in _BUCKET_FEATURE_KEYS:
        bucketized_feature = tft.bucketize(inputs[key], _FEATURE_BUCKET_COUNT[key])
        outputs[_transformed_name(key)] = tf.cast(bucketized_feature, tf.float32)

    # Convert strings to indices in a vocabulary
    # for key in _CATEGORICAL_FEATURE_KEYS:
    #     outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    # Convert the label strings to an index
    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[_LABEL_KEY],top_k = 5)

    return outputs


################
# Trainer code
################


# Import necessary libraries
from tensorflow import keras
from kerastuner.engine import base_tuner
import os
from pathlib import Path
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

# Constants from housing_constants_module_file
import housing_constant
_transformed_name = housing_constant.transformed_name

# defining a structure(NamedTuple) that can hold both the tuner tool (base_tuner) and the training instructions(fit_kwargs) together
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])
# defining the label
LABEL_KEY = "ocean_proximity_xf"




# 1. Load compressed dataset
def _gzip_reader_fn(filenames):  # helper function to load data 
  '''Load compressed dataset

  Args:
    filenames - filenames of TFRecords to load

  Returns:
    TFRecordDataset loaded from the filenames
  '''
#Loading the dataset. Specify the compression type since it is saved as `.gz`
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')




# 2. Data loading for model training and validation is performed in batches, and the loading is handled by the input_fn()
def _input_fn(file_pattern,     # helper function to load data and preprocessing 
              tf_transform_output,
              num_epochs = None,
              batch_size=32) -> tf.data.Dataset:
  '''Create batches of features and labels from TF Records

  Args:
    file_pattern - List of files or patterns of file paths containing Example records.
    tf_transform_output -The transform output graph from TensorFlow Transform (TF Transform) that contains information about how features have been transformed or preprocessed.
    num_epochs - Integer specifying the number of times to read through the dataset. 
            If None, cycles through the dataset forever.
    batch_size - An int representing the number of records to combine in a single batch.

  Returns:
    A dataset of dict elements, (or a tuple of dict elements and label). 
    Each dict maps feature keys to Tensor or SparseTensor objects.
  '''

  # Get feature specification based on transform output = This specification defines the structure and types of features after they have been transformed by TF Transform.
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  # Create batches of features and labels
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=LABEL_KEY)
  
  return dataset
# The input_fn returns a generator (a batched_features_dataset) that will supply
# data to the model one batch at a time.



# 3. Applying the preprocessing graph to model inputs
# This function bridges the gap between your trained model and its deployment for serving predictions. 
# It essentially defines how the model handles incoming requests containing new data points.

#Signatures specify how the model can be used for inference, including input and output formats.
def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer() #Load the preprocessing graph

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(housing_constant.LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec  #Parse the raw tf.Example records from the request.
        )

        transformed_features = model.tft_layer(parsed_features) #Apply the preprocessing transformation to raw data

        outputs = model(transformed_features) #Perform prediction with preprocessed data.
        return {"outputs": outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_eval = tf_transform_output.transform_features_layer()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    feature_spec = tf_transform_output.raw_feature_spec()
    parsed_features = tf.io.parse_example(serialized_tf_example, feature_spec)
    transformed_features = model.tft_layer_eval(parsed_features)
    # logging.info('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return transform_features_fn


def export_serving_model(tf_transform_output, model, output_dir):
  """Exports a keras model for serving.
  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    model: A keras model to export for serving.
    output_dir: A directory where the model will be exported to.
  """
  # The layer has to be saved to the model for keras tracking purpases.
  model.tft_layer = tf_transform_output.transform_features_layer()

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, tf_transform_output),
      'transform_features':
          _get_transform_features_signature(model, tf_transform_output),
  }

  model.save(output_dir, save_format='tf', signatures=signatures)



################################################################################################################################################################







# 4. DEFINING MODEL ARCHITECTURE
def build_model(hp):
  """Builds a Keras model for housing price prediction with hyperparameter tuning.

  Args:
      hp: Hyperparameters object from KerasTuner.

  Returns:
      A compiled Keras model.
  """

  input_features = []
  for x in housing_constant.NUMERIC_FEATURE_KEYS:
    input_features.append(tf.keras.Input(shape=(1,), name=_transformed_name(x)))

  for x in housing_constant.BUCKET_FEATURE_KEYS:
    input_features.append(tf.keras.Input(shape=(1,), name=_transformed_name(x)))  # Bucket features have shape (1,) after transformation

  # Concatenate all input layers
  concat = tf.keras.layers.concatenate(input_features)

  # Hidden layers
  first_layer = tf.keras.layers.Dense(units=hp.get('units_1'), activation='relu')(concat)
  dropout_layer = tf.keras.layers.Dropout(0.2)(first_layer)
  second_layer = tf.keras.layers.Dense(units=hp.get('units_2'), activation='relu')(dropout_layer)
  third_layer = tf.keras.layers.Dense(units=hp.get('units_3'), activation='relu')(second_layer)
  fourth_layer = tf.keras.layers.Dense(units=hp.get('units_4'), activation='relu')(third_layer)

  # Output layer
  output_layer = tf.keras.layers.Dense(5, activation='softmax')(third_layer)

  # Create the model
  model = tf.keras.Model(inputs=input_features, outputs=output_layer)

  # Compile the model
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.get('learning_rate') 
                                                ) ,loss='sparse_categorical_crossentropy',metrics=['categorical_accuracy'])
  # Print model summary for debugging purposes (optional)
  model.summary()

  return model




# 5. Training Orchestration
# run_fn is the main function that coordinates training. It calls functions in a specific order to:
# Load the transform graph (tf_transform_output)
# Prepare training and validation datasets using _input_fn with the transform graph
# Load hyperparameters from fn_args
# Build the model using build_model with loaded hyperparameters
# Train the model using model.fit with prepared datasets and callbacks
# Save the model with signatures using model.save


#The Trainer component will look for a run_fn() function in our module file and use
#the function as an entry point to execute the training process. 
#The module file needs The TFX Trainer Component to be accessible to the Trainer component

def run_fn(fn_args: FnArgs) -> None:
  """Defines and trains the model.
  Args:
    fn_args: Holds args as name/value pairs. fn_args: This is an object of type FnArgs that holds arguments as name/value pairs. 
    It contains various attributes such as paths to transform graphs, example datasets, hyperparameters, and model directories
    Refer here for the complete attributes: 
    https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
  """

#the run_fn function receives a set of arguments, including the transform
#graph, example datasets, and training parameters through the fn_args object
    
  # 5.1 Load transform output
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
  # Create batches of data good for 10 epochs
  train_set = _input_fn(fn_args.train_files, tf_transform_output, 10)
  val_set = _input_fn(fn_args.eval_files, tf_transform_output, 10)

  # 5.2 Load best hyperparameters
  hp = fn_args.hyperparameters.get('values')
    

  # 5.3 Build the model
  model = build_model(hp)

  #Callbacks
  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=log_dir, update_freq='batch')
    
  # 5.4 Train the model
  model.fit(x=train_set,validation_data=val_set , callbacks=[tensorboard_callback])

    
#Signatures specify how the model can be used for inference, including input and output formats.
  # signatures = {'serving_default': 
  #               _get_serve_tf_examples_fn(model,tf_transform_output).get_concrete_function(tf.TensorSpec(shape=[None],
  #                                                                                                        dtype=tf.string,name='examples'))}

    
  # 5.5 Save the model

  export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)  
  # model.save(fn_args.serving_model_dir, save_format='tf' ,signatures=signatures )  #,
  # model_name = "my_housing_model"
  # model_version = "0004_updated_units"
  # model_path = Path(model_name) / model_version   #another method to save the model
  # model.save(model_path, save_format="tf")





