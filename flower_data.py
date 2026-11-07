# -*- coding: utf-8 -*-
"""Flower image classification with a small Keras CNN.

Created on Sat Jan  5 01:09:37 2019

@author: Bhavesh

Running this file trains the network. Importing it only defines the helpers.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

CLASS_NAMES = ("daisy", "dandelion", "rose", "sunflower", "tulip")
DEFAULT_DATA_DIR = "D:/Image_data/flowers"
FEATURE_SIZE = (100, 100)
# The active CNN was written for this shape. It does not match FEATURE_SIZE.
CNN_INPUT_SHAPE = (32, 32, 3)
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42
MODEL_SEED = 2
EPOCHS = 30
BATCH_SIZE = 16


def image_data_extractor(image, size=FEATURE_SIZE):
    """Resize one image and return a flat pixel vector.

    The source mode is kept. RGB files become ``width * height * 3`` values
    and grayscale files become ``width * height`` values.
    """
    sample_image = Image.open(image)
    sample_image = sample_image.resize(size)
    # sample_image = sample_image.convert("L")
    return np.array(sample_image).flatten()


def class_glob_patterns(data_dir=DEFAULT_DATA_DIR):
    """Return the class globs used by the original script."""
    return [f"{data_dir}/{name}/*.jpg" for name in CLASS_NAMES]


def load_class_paths(data_dir=DEFAULT_DATA_DIR):
    """Map each class name to its ``*.jpg`` paths, in glob order."""
    patterns = class_glob_patterns(data_dir)
    return {name: glob.glob(pattern) for name, pattern in zip(CLASS_NAMES, patterns)}


def require_class_images(paths_by_class, data_dir):
    """Raise if any class folder contributed no JPEG paths."""
    missing = [name for name in CLASS_NAMES if not paths_by_class.get(name)]
    if missing:
        joined = ", ".join(missing)
        classes = ", ".join(CLASS_NAMES)
        raise FileNotFoundError(
            f"No .jpg images for {joined} under {data_dir}. "
            f"Expected one folder per class: {classes}."
        )


def build_labeled_frame(paths_by_class):
    """Stack flattened images and integer labels in class order.

    Labels are daisy 0, dandelion 1, rose 2, sunflower 3, tulip 4.
    """
    frames = []
    labels = []
    for index, name in enumerate(CLASS_NAMES):
        images = paths_by_class[name]
        rows = [list(image_data_extractor(image)) for image in images]
        frames.append(pd.DataFrame(rows))
        labels.append(pd.DataFrame(np.repeat(index, len(images))))
    combined = pd.concat(frames, axis=0)
    target = pd.concat(labels, axis=0)
    target.columns = ["Label"]
    return pd.concat([combined, target], axis=1)


def prepare_train_test(
    final_df,
    test_size=TEST_SIZE,
    random_state=SPLIT_RANDOM_STATE,
    shuffle_rows=True,
):
    """Shuffle, hold out the test rows, and scale pixels from 0–255 to 0–1."""
    if shuffle_rows:
        final_df = shuffle(final_df)
    target = final_df[["Label"]]
    features = final_df.drop("Label", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    return np.array(x_train) / 255, np.array(x_test) / 255, y_train, y_test


def reshape_for_cnn(samples, shape=CNN_INPUT_SHAPE):
    """Reshape flat image rows into the CNN's ``(32, 32, 3)`` input."""
    expected = shape[0] * shape[1] * shape[2]
    width = int(samples.shape[1])
    if width != expected:
        feature_count = FEATURE_SIZE[0] * FEATURE_SIZE[1] * 3
        raise ValueError(
            f"Cannot reshape image vectors of length {width} into {shape} "
            f"({expected} values). image_data_extractor resizes to "
            f"{FEATURE_SIZE[0]}x{FEATURE_SIZE[1]} and keeps the source mode, "
            f"so an RGB file has {feature_count} values. The original script "
            "used both sizes, and training still stops on this mismatch."
        )
    return samples.reshape((samples.shape[0],) + tuple(shape))


def _keras_symbols():
    """Import Keras pieces, including the 2019 ``np_utils`` path when present."""
    try:
        from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
        from keras.models import Sequential
    except ImportError:
        from tensorflow.keras.layers import (
            Conv2D,
            Dense,
            Dropout,
            Flatten,
            Input,
            MaxPooling2D,
        )
        from tensorflow.keras.models import Sequential
    try:
        from keras.utils import np_utils

        to_categorical = np_utils.to_categorical
    except ImportError:
        try:
            from keras.utils import to_categorical
        except ImportError:
            from tensorflow.keras.utils import to_categorical
    return Sequential, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, to_categorical


def set_model_seed(seed=MODEL_SEED):
    """Seed TensorFlow the way the original script seeded it."""
    import tensorflow as tf

    if hasattr(tf, "set_random_seed"):
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)


def build_model():
    """Build the original five-class CNN. Weights are not loaded from disk."""
    Sequential, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, _to_categorical = (
        _keras_symbols()
    )
    model = Sequential()
    model.add(Input(shape=CNN_INPUT_SHAPE))
    model.add(Conv2D(64, kernel_size=5, padding="same", activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(150, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(600, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def plot_history(final_model):
    """Plot training and validation accuracy and loss.

    Older Keras stored ``acc`` and ``val_acc``. Current TensorFlow stores
    ``accuracy`` and ``val_accuracy``.
    """
    import matplotlib.pyplot as plt

    record = final_model.history
    accuracy = record["acc"] if "acc" in record else record["accuracy"]
    val_accuracy = record["val_acc"] if "val_acc" in record else record["val_accuracy"]
    loss = record["loss"]
    val_loss = record["val_loss"]
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def train(x_train, x_test, y_train, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2):
    """Encode labels, reshape to the CNN input, and fit for ``epochs`` epochs."""
    x_train = reshape_for_cnn(np.asarray(x_train))
    x_test = reshape_for_cnn(np.asarray(x_test))
    _sequential, _conv, _pool, _drop, _flat, _dense, _input, to_categorical = _keras_symbols()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    set_model_seed(MODEL_SEED)
    model = build_model()
    model.summary()
    final_model = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=verbose,
    )
    plot_history(final_model)
    return final_model


def main(argv=None):
    """Load the flower folders and train. Default data dir is unchanged."""
    parser = argparse.ArgumentParser(
        description="Train the flower CNN on daisy, dandelion, rose, sunflower, and tulip folders."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("FLOWER_DATA_DIR", DEFAULT_DATA_DIR),
        help=(
            "Directory that contains the five class folders "
            f"(default: {DEFAULT_DATA_DIR}, or FLOWER_DATA_DIR when set)."
        ),
    )
    args = parser.parse_args(argv)
    paths = load_class_paths(args.data_dir)
    require_class_images(paths, args.data_dir)
    frame = build_labeled_frame(paths)
    x_train, x_test, y_train, y_test = prepare_train_test(frame)
    train(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
