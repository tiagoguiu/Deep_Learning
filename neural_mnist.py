import tensorflow as tf

print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss') <0.01): 
      print("\nAtingiu loss < 0.1, cancelando treinamento")
      self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.mnist

(training_x, training_y), (test_images, test_labels) = mnist.load_data()
training_images = training_x /255.0

test_images = test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(64, activation=tf.nn.relu),
  tf.keras.layers.Dense(32, activation=tf.nn.relu),

  tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training_images, training_y, epochs=10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])

print(test_labels[0])

# Utilizei uma rede com 4 camadas no modelo relu e uma camada modelo softMax
# A acuracia geral foi de 98%, a maior foi 99.34% foi o maior que consegui