import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'target'
FEATURE_KEY = ['calc','cond','gravity','osmo','ph','urea']

NUM_EPOCHS = 10

def transformed_name(key):
    """Renaming transformed features and strip transform"""
    key=key.strip()
    return key + "_xf"

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=10,
    restore_best_weights=True,
    baseline=0.8
)


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output,num_epochs, batch_size=64):

    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    ).repeat(50)

    return dataset


def model_builder(hp):
    num_hidden_layers = hp.Choice(
        "num_hidden_layers", values=[1, 2]
    )
    dense_units = hp.Int(
        "dense_units", min_value=32, max_value=256, step=32
    )
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
    )
    learning_rate = hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4]
    )
    
    input_features = []

    for feature in FEATURE_KEY:
        input_features.append(
            layers.Input(shape=(1,), name=transformed_name(feature),dtype=tf.float32)
        )

    concatenate = layers.concatenate(input_features)
    x = layers.Dense(dense_units, activation=tf.nn.relu)(concatenate)  

    for _ in range(num_hidden_layers):
        x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
    
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["binary_accuracy"],
    )
    
    return model


def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_set = input_fn(
        fn_args.train_files, tf_transform_output, NUM_EPOCHS
    )
    eval_set = input_fn(
        fn_args.eval_files, tf_transform_output, NUM_EPOCHS
    )

    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp),
        objective=kt.Objective('binary_accuracy', direction='max'),
        max_epochs=NUM_EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kt_hyperband",
    )
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_callback],
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
