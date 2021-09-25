import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.01):
      print("\nAtingiu loss < 0.1, cancelando treinamento")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(16, activation=tf.nn.relu),

  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])

print(test_labels[0])

# A acuracia geral foi de 88%, a maior foi 91% foi o maior que consegui
# Testei varias combinações de multiplas camadas, e com muitos neuronios, mas meu melhor resultado foi esse