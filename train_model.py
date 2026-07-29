"""
Train a cat vs dog classifier using transfer learning (MobileNetV2).
Works well even with a few hundred images per class, unlike a from-scratch CNN.

Folder structure expected:
dataset/
    cats/   -> all cat images (.jpg/.png)
    dogs/   -> all dog images (.jpg/.png)
"""

import tensorflow as tf

IMG_SIZE = (160, 160)      # MobileNetV2's native size (better than 128x128 for this)
BATCH_SIZE = 16
EPOCHS_HEAD = 10           # train just the new head
EPOCHS_FINE_TUNE = 8       # then unfreeze part of the base and fine-tune

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)

class_names = train_data.class_names   # e.g. ['cats', 'dogs'] -> index 0/1
print("Class order (index -> label):", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# 2. Data augmentation (helps a lot with small datasets)
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
])

# ---------------------------------------------------------------------------
# 3. Build model with MobileNetV2 base (pretrained on ImageNet)
# ---------------------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze for initial training

# MobileNetV2 expects inputs in [-1, 1], this preprocess layer handles that
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

print("\n--- Phase 1: training the classification head ---")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stop],
)

# ---------------------------------------------------------------------------
# 4. Fine-tune: unfreeze the top layers of the base model and train with a
#    much lower learning rate. This squeezes out extra accuracy.
# ---------------------------------------------------------------------------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # unfreeze last 30 layers only
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Phase 2: fine-tuning top layers of MobileNetV2 ---")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[early_stop],
)

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
model.save("cat_dogs_model.keras")
print("\nModel saved as cat_dogs_model.keras")
print("Class order used by the model (0 -> 1):", class_names)