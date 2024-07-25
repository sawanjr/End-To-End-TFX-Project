
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
