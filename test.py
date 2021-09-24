import os

metrics_file = "metrics.csv"
metrics_delimiter = ","


def test_data_zip_does_not_exist():
    assert not os.path.exists("data.zip"), "data.zip should not exist"


def test_data_directory_does_not_exist():
    assert not os.path.exists("data"), "data directory should not exist"


def test_saved_model_does_not_exist():
    assert not os.path.isfile("model.pth"), "Saved model file should not exist"


def get_accuracy_indices(metrics):
    header_cols = metrics[0].strip().split(metrics_delimiter)
    val_accuracy_idx = -1
    dog_accuracy_idx = -1
    cat_accuracy_idx = -1
    for idx, h in enumerate(header_cols):
        if h == "val_accuracy":
            val_accuracy_idx = idx
        elif h == "dog_accuracy":
            dog_accuracy_idx = idx
        elif h == "cat_accuracy":
            cat_accuracy_idx = idx
    return val_accuracy_idx, dog_accuracy_idx, cat_accuracy_idx


def test_overall_acurracy_greater_than_70_percent():
    metrics = open(metrics_file).readlines()
    val_accuracy_idx, _, _ = get_accuracy_indices(metrics)
    assert val_accuracy_idx >= 0, "Get val_accuracy column index"
    data = metrics[-1].strip().split(metrics_delimiter)
    assert float(data[val_accuracy_idx]) > 70.0, "Validation accuracy is greater than 70%"


def test_dog_acurracy_greater_than_70_percent():
    metrics = open(metrics_file).readlines()
    _, dog_accuracy_idx, _ = get_accuracy_indices(metrics)
    assert dog_accuracy_idx >= 0, "Get dog_accuracy column index"
    data = metrics[-1].strip().split(metrics_delimiter)
    assert float(data[dog_accuracy_idx]) > 70.0, "Dog Validation accuracy is greater than 70%"


def test_cat_acurracy_greater_than_70_percent():
    metrics = open(metrics_file).readlines()
    _, _, cat_accuracy_idx = get_accuracy_indices(metrics)
    assert cat_accuracy_idx >= 0, "Get cat_accuracy column index"
    data = metrics[-1].strip().split(metrics_delimiter)
    assert float(data[cat_accuracy_idx]) > 70.0, "Cat Validation accuracy is greater than 70%"
