import tensorflow as tf
import zipfile
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

base_dir = '/tmp/fire-smoke'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_fire_dir = os.path.join(train_dir, 'fire')
train_smoke_dir = os.path.join(train_dir, 'smoke')

validation_fire_dir = os.path.join(validation_dir, 'fire')
validation_smoke_dir = os.path.join(validation_dir, 'smoke')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_steps=50,
                              verbose=2)