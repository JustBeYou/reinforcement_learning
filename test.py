from gridsearch import gridsearch
import numpy as np

def test_target(a=5, b=3, c=7, d=50):
    x = -55
    return x**3 * a + x**2 * b + x * c + d

grid = {
    "a": np.arange(-25, 30, 1),
    "b": np.arange(-25, 30, 1),
    "c": np.arange(-25, 30, 1),
    "d": np.arange(-25, 30, 1),
}

print(test_target(-25,  24, -25,  24))
r = gridsearch(grid, test_target, workers=4)
print (r)
