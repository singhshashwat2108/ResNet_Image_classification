from utils import load_data, plot_history
from utils import evaluate_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import os

# Use all CPU cores
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)

# Optional: force CPU (no GPU confusion)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Num CPUs used:", tf.config.threading.get_intra_op_parallelism_threads())


x_train, x_test, y_train, y_test = load_data()

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)

def residual_block(x, filters):
    shortcut = x

    # Main path
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 🔥 FIX: match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)

    # Skip connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


inputs = layers.Input(shape=(32,32,3))

x = layers.Conv2D(32, (3,3), padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Residual blocks
x = residual_block(x, 32)
x = residual_block(x, 32)

x = layers.MaxPooling2D()(x)

x = residual_block(x, 64)
x = residual_block(x, 64)

x = layers.GlobalAveragePooling2D()(x)

outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[lr_scheduler]
)

plot_history(history, "ResNet Model")
evaluate_model(model, x_test, y_test)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Final Test Accuracy:", accuracy)