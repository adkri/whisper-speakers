"""Test"""

import mlx.core as mx
import mlx.nn as nn


a = mx.array([1, 2, 3])

print(a)
print(a.shape)
print(a.dtype)

b = mx.array([[1, 2, 3], [4, 5, 6]])


c = a + b
print(c)
