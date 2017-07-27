# coding=utf-8
"""Load proper subclass of Model based on its name."""
# noinspection PyUnresolvedReferences
from . import implementations
from .model import Model


def get_model(model_name, batch_size, num_classes):
    """Selects the subclass of Model with the specified name."""
    class_selection_dict = {}
    for cls in Model.__subclasses__():
        name = cls.get_name()
        if name in class_selection_dict:
            raise RuntimeError(
                "Models '{}' and '{}' have same name: '{}'".format(
                    cls.__name__, class_selection_dict[name].__name__, name))
        class_selection_dict[name] = cls
    try:
        model_class = class_selection_dict[model_name]
    except KeyError as e:
        e.args = e.args or ('',)
        e.args += ("No model with name '{}'.".format(model_name),)
        raise
    return model_class(batch_size, num_classes)
