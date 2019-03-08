import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

X, y = make_circles(n_samples=1000,
                    noise=.1,
                    factor=.2,
                    random_state=0)

