import tensorflow as tf
import zipfile
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


DESIRED_ACCURACY = 0.99

!wget - -no-check-certificate \
    "https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip" - O "/tmp/fire-smoke.zip"

zip_ref = zipfile.ZipFile("/tmp/fire-smoke.zip", 'r')
zip_ref.extractall("/tmp/")
zip_ref.close()

# Conjunto Fumaça-Fogo
!mkdir - p / tmp/S-F/{Train, Test}/
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Train/Smoke" "/tmp/S-F/Train/Smoke"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Test/Smoke" "/tmp/S-F/Test/Smoke"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Train/Fire" "/tmp/S-F/Train/Fire"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Test/Fire" "/tmp/S-F/Test/Fire"

# Conjunto Neutro-Fogo
!mkdir - p / tmp/N-F/{Train, Test}/
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Train/Neutral" "/tmp/N-F/Train/Neutral"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Test/Neutral" "/tmp/N-F/Test/Neutral"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Train/Fire" "/tmp/N-F/Train/Fire"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Test/Fire" "/tmp/N-F/Test/Fire"

# Conjunto Neutro-Fumaça
!mkdir - p / tmp/N-S/{Train, Test}/
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Train/Neutral" "/tmp/N-S/Train/Neutral"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Test/Neutral" "/tmp/N-S/Test/Neutral"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Train/Smoke" "/tmp/N-S/Train/Smoke"
!ln - sf "/tmp/FIRE-SMOKE-DATASET/Test/Smoke" "/tmp/N-S/Test/Smoke"

pre_trained_model = InceptionV3(input_shape=(
    150, 150, 3), include_top=False, weights=None)

base_dir = '/tmp/fire-smoke'

pre_trained_model.load_weights(base_dir)
for layer in pre_trained_model.layers:
    layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_fire_dir = os.path.join(train_dir, 'fire')
train_smoke_dir = os.path.join(train_dir, 'smoke')

validation_fire_dir = os.path.join(validation_dir, 'fire')
validation_smoke_dir = os.path.join(validation_dir, 'smoke')

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size=20, class_mode='binary', target_size=(150, 150))

validation_generator = test_datagen.flow_from_directory(
    validation_dir, batch_size=20, class_mode='binary', target_size=(150, 150))

history = model.fit(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=20, validation_steps=50, verbose=2)
