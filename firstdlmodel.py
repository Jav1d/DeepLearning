import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

X, y = make_circles(n_samples=1000,
                    noise=.1,
                    factor=.2,
                    random_state=0)

plt.figure(figsize=(5, 5))
plt.plot(X[y == 0, 0], X[y == 0, 1], 'ob', alpha=0.5)
plt.plot(X[y == 1, 0], X[y == 1, 1], 'xr', alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(['0', '1'])

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20)
