# coding=utf-8
"""Selects dataset from specified dataset type."""
from .cifar10_data import CIFAR10Data

# MUST UPDATE THIS DICT TO MAKE NEW DATASET CLASSES AVAILABLE
CLASS_SELECTION_DICT = {
    'cifar10': CIFAR10Data
}


def get_dataset(dataset_name, data_dir, overwrite):
    """Selects dataset from specified dataset name."""
    try:
        dataset_class = CLASS_SELECTION_DICT[dataset_name]
    except KeyError as e:
        e.args = e.args or ('',)
        e.args += ("Dataset '{}' not available.".format(dataset_name))
        raise
    return dataset_class(data_dir, overwrite)
