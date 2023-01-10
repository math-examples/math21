from __future__ import absolute_import
from __future__ import print_function
from builtins import map
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from m21py.ad import *
from m21py.m_convert import *


# x = npr.randn(2,3)
# x = np.array([[4, 0, 2, 1],
#               [3, 2, 2, 0]])
x = np.array([0.17640523, 0.04001572, 0.0978738])
print(x)

import time

t = time.time()

def tanh_like_dn(x, n):
    x_raw = math21_python_tensor_create_from_ndarray(x)
    # y_raw = math21_test_ad_logsumexp_like(x_raw, n)
    y_raw = math21_test_ad_logsumexp_like(x_raw)
    if y_raw.type == 0:
        raise ValueError("emtpy")
    math21_tensor_destroy(x_raw)
    y = math21_python_convert_tensor_to_ndarray(y_raw)
    return y

print(tanh_like_dn(x, 0))

x = np.linspace(-7, 7, 200)

dt, _ = math21_python_convert_type_to_dtype(m21type.m21_type_NumR.value)
x = x.astype(dt)


plt.plot(
    x, tanh_like_dn(x, 0),  # zero derivative
    x, tanh_like_dn(x, 1),  # first derivative
    x, tanh_like_dn(x, 2),  # second derivative
    x, tanh_like_dn(x, 3),  # third derivative
    x, tanh_like_dn(x, 4),  # fourth derivative
    x, tanh_like_dn(x, 5),  # fifth derivative
    x, tanh_like_dn(x, 6)  # sixth derivative
)

plt.axis('off')
plt.savefig("tanh.png")
plt.show()

print('time cost {}'.format(time.time() - t))