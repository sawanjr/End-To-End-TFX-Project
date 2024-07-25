# Import necessary libraries
from kerastuner.engine import base_tuner
import kerastuner as kt
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import TFTransformOutput


# Constants from housing_constants_module_file
import housing_constant
_transformed_name = housing_constant.transformed_name


TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

LABEL_KEY = "ocean_proximity_xf"


def _gzip_reader_fn(filenames):
  '''Load compressed dataset
  
  Args:
    filenames - filenames of TFRecords to load

  Returns:
    TFRecordDataset loaded from the filenames
  '''

  # Loading the dataset. Specify the compression type since it is saved as `.gz`
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
  '''Create batches of features and labels from TF Records

  Args:
    file_pattern - List of files or patterns of file paths containing Example records.
    tf_transform_output - transform output graph
    num_epochs - Integer specifying the number of times to read through the dataset. 
            If None, cycles through the dataset forever.
    batch_size - An int representing the number of records to combine in a single batch.

  Returns:
    A dataset of dict elements, (or a tuple of dict elements and label). 
    Each dict maps feature keys to Tensor or SparseTensor objects.
  '''

  # Get feature specification based on transform output
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

def build_model(hp):
    input_features = []
    for x in housing_constant.NUMERIC_FEATURE_KEYS:
        input_features.append(tf.keras.Input(shape=(1,), name=_transformed_name(x)))

    for x in housing_constant.BUCKET_FEATURE_KEYS:
        input_features.append(tf.keras.Input(shape=(1,), name=_transformed_name(x)))  # Bucket features have shape (1,) after transformation

    # Concatenate all input layers
    concat = tf.keras.layers.concatenate(input_features)

    # Hidden layers
    first_layer = tf.keras.layers.Dense(units=hp.Int('units_1', min_value=100, max_value=500, step=5), activation='relu')(concat)
    dropout_layer = tf.keras.layers.Dropout(0.1)(first_layer)
    second_layer = tf.keras.layers.Dense(units=hp.Int('units_2', min_value=125, max_value=500, step=25), activation='relu')(dropout_layer)
    third_layer = tf.keras.layers.Dense(units=hp.Int('units_3', min_value=150, max_value=500, step=25), activation='relu')(second_layer)
    fourth_layer = tf.keras.layers.Dense(units=hp.Int('units_4', min_value=100, max_value=200, step=25), activation='relu')(third_layer)


    # Output layer
    output_layer = tf.keras.layers.Dense(units=5, activation='softmax')(fourth_layer)

    # Create the model
    model = tf.keras.Model(inputs=input_features, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    
    return model



########################################################################################################






def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.
    Args:
        fn_args: Holds args as name/value pairs.

    Returns:
        A BaseTuner object that will be used for tuning.
    """ 

    # Initialize the tuner
    tuner = kt.Hyperband(build_model,
                         objective='val_loss',
                         max_epochs=15,
                         factor=3,
                         directory=fn_args.working_dir,
                         project_name='housing_tuning')

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    # Use _input_fn() to extract input features and labels from the train and val set
    train_set = _input_fn(fn_args.train_files, tf_transform_output)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output)


    return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
  )
