from cec2017 import functions as functions
import numpy as np


f = functions.f3
dimension = 30
for i in range(0, 10):
    x = np.random.uniform(low=-100, high=100, size=dimension)
    y = f([x])
    print(y)
    y = f([x])[0]
    print(f"f5({x[0]:.2f}, {x[1]:.2f}, ...) = {y:.2f}")