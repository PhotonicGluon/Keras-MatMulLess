"""
Use the ``ternary_mul`` op in python.
"""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

ternary_mul_ops = load_library.load_op_library(resource_loader.get_path_to_datafile("_ternary_mul_ops.so"))
ternary_mul = ternary_mul_ops.ternary_mul
