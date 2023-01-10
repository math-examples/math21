from __future__ import absolute_import
from __future__ import print_function
from builtins import map
import numpy as np
import matplotlib.pyplot as plt
from m21py.ad import *

fun = lambda x: math21_test_c_ad_sin_dn(x, 0)
d_fun = lambda x: math21_test_c_ad_sin_dn(x, 1)  # First derivative
dd_fun = lambda x: math21_test_c_ad_sin_dn(x, 2)  # Second derivative

x = np.linspace(-10, 10, 100)
plt.plot(x, list(map(fun, x)), x, list(map(d_fun, x)), x, list(map(dd_fun, x)))

plt.xlim([-10, 10])
plt.ylim([-1.2, 1.2])
# plt.axis('off')
plt.savefig("sinusoid.png")
# plt.show()
plt.clf()

fun = lambda x: math21_test_c_ad_sin_taylor_appr_dn(x, 0)
d_fun = lambda x: math21_test_c_ad_sin_taylor_appr_dn(x, 1)  # First derivative
dd_fun = lambda x: math21_test_c_ad_sin_taylor_appr_dn(x, 2)  # Second derivative

x = np.linspace(-10, 10, 100)
plt.plot(x, list(map(fun, x)), x, list(map(d_fun, x)), x, list(map(dd_fun, x)))

plt.xlim([-10, 10])
plt.ylim([-1.2, 1.2])
# plt.axis('off')
plt.savefig("sinusoid_taylor.png")
plt.show()
plt.clf()
