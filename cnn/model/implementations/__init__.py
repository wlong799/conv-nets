# coding=utf-8
"""Package contains all implemented Model subclasses. Package automatically
finds and imports any subclass modules once they are written (i.e. no need
for manually adding their import statement to this file)."""
import glob
import os

_all_module_filenames = glob.glob(os.path.dirname(__file__) + "/*.py")
_model_module_filenames = [filename for filename in _all_module_filenames
                           if not filename.endswith('__init__.py')]
_model_module_names = [os.path.splitext(os.path.basename(filename))[0] for
                       filename in _model_module_filenames]
__all__ = _model_module_names
from . import *
