# coding=utf-8
"""Selects dataset from specified dataset type."""
from .datasets import BasicDataset, ConfigDataset
from .cifar10_data import CIFAR10Data

# MUST UPDATE THIS DICT TO MAKE NEW DATASET CLASSES AVAILABLE
CLASS_SELECTION_DICT = {
    'cifar10': CIFAR10Data
}


def get_dataset(dataset_name, data_dir, overwrite, config=None):
    """Selects dataset from specified dataset name."""
    try:
        dataset_class = CLASS_SELECTION_DICT[dataset_name]
    except KeyError as e:
        e.args = e.args or ('',)
        e.args += ("Dataset '{}' not available.".format(dataset_name))
        raise
    if issubclass(dataset_class, BasicDataset):
        return dataset_class(data_dir, overwrite)
    elif issubclass(dataset_class, ConfigDataset):
        return dataset_class(data_dir, overwrite, config)
    else:
        raise ValueError(
            "Dataset '{}' must subclass either BasicDataset or ConfigDataset.")
