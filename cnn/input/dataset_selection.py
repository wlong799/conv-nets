# coding=utf-8
"""Load proper dataset implementation based on its name."""
# noinspection PyUnresolvedReferences
from . import implementations
from .datasets import BasicDataset, ConfigDataset


def get_dataset(dataset_name, data_dir, overwrite, config):
    """Selects the subclass of Model with the specified name."""
    subclasses = []
    subclasses.extend(BasicDataset.__subclasses__())
    subclasses.extend(ConfigDataset.__subclasses__())
    class_selection_dict = {}
    for cls in subclasses:
        name = cls.get_name()
        if name in class_selection_dict:
            raise RuntimeError(
                "Datasets '{}' and '{}' have same name: '{}'".format(
                    cls.__name__, class_selection_dict[name].__name__, name))
        class_selection_dict[name] = cls
    try:
        model_class = class_selection_dict[dataset_name]
    except KeyError as e:
        e.args = e.args or ('',)
        e.args += ("No dataset with name '{}'.".format(dataset_name),)
        raise
    if issubclass(model_class, BasicDataset):
        return model_class(data_dir, overwrite)
    elif issubclass(model_class, ConfigDataset):
        return model_class(data_dir, overwrite, config)
    else:
        raise RuntimeError("'{}' is not a subclass of BasicDataset or"
                           "ConfigDataset.".format(model_class.__name__))
