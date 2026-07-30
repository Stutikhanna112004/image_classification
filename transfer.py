# convert_to_tflite.py
import tensorflow as tf

model = tf.keras.models.load_model("backend/cat_dogs_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # shrinks size + memory further
tflite_model = converter.convert()

with open("cat_dogs_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved cat_dogs_model.tflite")