import numpy as np
a = np.array([0.1, 0.2, 0.3, 0.4])
b = np.array([0.4, 0.3, 0.2, 0.1])
c = np.array([0., 0., 1., 0.])
print(np.multiply(c, a, 1. / b))

