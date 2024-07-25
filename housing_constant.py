
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
