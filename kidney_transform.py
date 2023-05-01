import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'target'
FEATURE_KEY = ['calc','cond','gravity','osmo','ph','urea']
def transformed_name(key):
    """Renaming transformed features and strip transform"""
    key=key.strip()
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
        
    outputs = {}
    # melakukan scale
    for i in FEATURE_KEY:
      outputs[transformed_name(i)] =tf.cast( tft.scale_to_z_score(inputs[i]), tf.float32)

    # merubah tipe data
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
