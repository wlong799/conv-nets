# coding=utf-8
"""Selects model from specified model type."""
from .simple_model import SimpleModel

# MUST UPDATE THIS DICT TO MAKE NEW MODEL CLASSES AVAILABLE
CLASS_SELECTION_DICT = {
    'simple': SimpleModel
}


def get_model(model_type, batch_size, num_classes):
    """Selects model from specified model type."""
    try:
        model_class = CLASS_SELECTION_DICT[model_type]
    except KeyError as e:
        e.args = e.args or ('',)
        e.args += ("Model type '{}' not available.".format(model_type))
        raise
    return model_class(batch_size, num_classes)
