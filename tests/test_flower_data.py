"""Behavior checks for the flower data helpers. Training is not the default path."""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.model_selection import train_test_split

import flower_data
from flower_data import (
    CLASS_NAMES,
    CNN_INPUT_SHAPE,
    DEFAULT_DATA_DIR,
    FEATURE_SIZE,
    build_labeled_frame,
    class_glob_patterns,
    image_data_extractor,
    load_class_paths,
    main,
    plot_history,
    prepare_train_test,
    reshape_for_cnn,
    train,
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _write_solid(path, mode, color, size):
    Image.new(mode, size, color).save(path)


def _write_class_jpegs(root, counts):
    root.mkdir(parents=True, exist_ok=True)
    for name, count in zip(CLASS_NAMES, counts):
        folder = root / name
        folder.mkdir()
        for index in range(count):
            _write_solid(folder / f"{index}.jpg", "RGB", (index * 20, 10, 30), (6, 8))
    return root


def test_default_glob_patterns_match_original_paths():
    assert DEFAULT_DATA_DIR == "D:/Image_data/flowers"
    assert class_glob_patterns() == [
        "D:/Image_data/flowers/daisy/*.jpg",
        "D:/Image_data/flowers/dandelion/*.jpg",
        "D:/Image_data/flowers/rose/*.jpg",
        "D:/Image_data/flowers/sunflower/*.jpg",
        "D:/Image_data/flowers/tulip/*.jpg",
    ]


def test_import_does_not_fit_a_model():
    assert not hasattr(flower_data, "final_model")
    assert callable(main)


def test_rgb_extractor_flattens_resized_image(tmp_path):
    path = tmp_path / "solid.png"
    _write_solid(path, "RGB", (10, 20, 30), (7, 9))
    flat = image_data_extractor(path)
    assert flat.shape == (FEATURE_SIZE[0] * FEATURE_SIZE[1] * 3,)
    assert flat.dtype == np.uint8
    assert list(flat[:3]) == [10, 20, 30]
    assert set(np.unique(flat[0::3])) == {10}
    assert set(np.unique(flat[1::3])) == {20}
    assert set(np.unique(flat[2::3])) == {30}


def test_grayscale_extractor_is_not_expanded_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    _write_solid(path, "L", 40, (4, 4))
    flat = image_data_extractor(path)
    assert flat.shape == (FEATURE_SIZE[0] * FEATURE_SIZE[1],)
    assert int(flat.max()) == 40
    assert int(flat.min()) == 40


def test_labeled_frame_follows_class_order_and_glob_order(tmp_path, monkeypatch):
    root = _write_class_jpegs(tmp_path / "flowers", (2, 1, 3, 1, 2))
    paths = load_class_paths(str(root))
    seen = []

    def fake_extractor(image):
        seen.append(os.path.abspath(image))
        return np.arange(4, dtype=np.uint8)

    monkeypatch.setattr(flower_data, "image_data_extractor", fake_extractor)
    frame = build_labeled_frame(paths)
    expected_paths = [os.path.abspath(path) for name in CLASS_NAMES for path in paths[name]]
    expected_labels = [index for index, name in enumerate(CLASS_NAMES) for _path in paths[name]]
    assert seen == expected_paths
    assert list(frame["Label"]) == expected_labels
    assert len(frame) == sum(len(paths[name]) for name in CLASS_NAMES)
    assert list(frame.columns[:4]) == [0, 1, 2, 3]


def test_prepare_train_test_uses_holdout_and_scales_pixels():
    frame = pd.DataFrame(
        {
            0: [0, 255, 128, 64, 32],
            1: [255, 0, 64, 128, 32],
            "Label": [0, 1, 2, 3, 4],
        }
    )
    features = frame.drop(columns=["Label"])
    target = frame[["Label"]]
    expected_x_train, expected_x_test, expected_y_train, expected_y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = prepare_train_test(frame, shuffle_rows=False)
    assert len(x_train) == 4
    assert len(x_test) == 1
    np.testing.assert_allclose(x_train, np.array(expected_x_train) / 255)
    np.testing.assert_allclose(x_test, np.array(expected_x_test) / 255)
    pd.testing.assert_frame_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(y_test, expected_y_test)
    assert x_train.max() <= 1
    assert y_train["Label"].max() > 1


def test_prepare_train_test_shuffles_before_splitting(monkeypatch):
    frame = pd.DataFrame({0: range(10), "Label": range(10)})
    calls = []

    def reverse(df):
        calls.append(True)
        return df.iloc[::-1]

    monkeypatch.setattr(flower_data, "shuffle", reverse)
    reversed_frame = frame.iloc[::-1]
    features = reversed_frame.drop(columns=["Label"])
    target = reversed_frame[["Label"]]
    _x_train, _x_test, expected_y_train, expected_y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )
    _x_train, _x_test, y_train, y_test = prepare_train_test(frame)
    assert calls == [True]
    pd.testing.assert_frame_equal(y_train, expected_y_train)
    pd.testing.assert_frame_equal(y_test, expected_y_test)


def test_reshape_rejects_rgb_vectors_from_the_feature_size():
    width = FEATURE_SIZE[0] * FEATURE_SIZE[1] * 3
    samples = np.zeros((2, width))
    with pytest.raises(ValueError, match="Cannot reshape"):
        reshape_for_cnn(samples)
    assert CNN_INPUT_SHAPE == (32, 32, 3)
    assert width != CNN_INPUT_SHAPE[0] * CNN_INPUT_SHAPE[1] * CNN_INPUT_SHAPE[2]


def test_main_missing_class_names_the_folder(tmp_path):
    root = tmp_path / "flowers"
    (root / "daisy").mkdir(parents=True)
    _write_solid(root / "daisy" / "0.jpg", "RGB", (1, 2, 3), (4, 4))
    with pytest.raises(FileNotFoundError, match="dandelion"):
        main(["--data-dir", str(root)])


def test_main_on_sample_jpegs_stops_on_the_shape_mismatch(tmp_path):
    root = _write_class_jpegs(tmp_path / "flowers", (1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="Cannot reshape"):
        main(["--data-dir", str(root)])


@pytest.mark.parametrize(
    "history",
    [
        {"acc": [0.2], "val_acc": [0.1], "loss": [1.5], "val_loss": [1.6]},
        {"accuracy": [0.4], "val_accuracy": [0.3], "loss": [1.1], "val_loss": [1.2]},
    ],
)
def test_plot_history_accepts_legacy_and_current_metric_names(history):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_history(SimpleNamespace(history=history))
    assert len(plt.get_fignums()) == 2
    plt.close("all")


def test_train_one_epoch_on_vectors_that_match_the_cnn():
    pytest.importorskip("tensorflow")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    width = CNN_INPUT_SHAPE[0] * CNN_INPUT_SHAPE[1] * CNN_INPUT_SHAPE[2]
    x_train = np.zeros((6, width), dtype=np.float64)
    x_test = np.zeros((4, width), dtype=np.float64)
    y_train = pd.DataFrame({"Label": [0, 1, 2, 3, 4, 1]})
    y_test = pd.DataFrame({"Label": [4, 0, 2, 3]})
    history = train(x_train, x_test, y_train, y_test, epochs=1, batch_size=2, verbose=0)
    record = history.history
    assert len(record["loss"]) == 1
    assert "acc" in record or "accuracy" in record
    assert "val_loss" in record
    conv = [layer for layer in history.model.layers if layer.__class__.__name__ == "Conv2D"]
    dense = [layer for layer in history.model.layers if layer.__class__.__name__ == "Dense"]
    dropouts = [layer for layer in history.model.layers if layer.__class__.__name__ == "Dropout"]
    assert [layer.filters for layer in conv] == [64, 128, 150]
    assert [tuple(layer.kernel_size) for layer in conv] == [(5, 5), (2, 2), (2, 2)]
    assert [layer.units for layer in dense] == [500, 600, 5]
    assert dense[-1].activation.__name__ == "softmax"
    assert len(dropouts) == 5
    assert all(layer.rate == 0.2 for layer in dropouts)
