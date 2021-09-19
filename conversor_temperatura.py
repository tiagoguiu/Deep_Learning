import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([0.0,  0.5 , 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([3.2, 4.1, 5.0, 6.8, 8.6, 10.4], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([38]))

#resultado [[72.3974]]