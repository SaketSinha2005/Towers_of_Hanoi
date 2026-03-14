import tensorflow as tf
import numpy as np

model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("ml/model.h5")
    return model


def predict(model, volume):

    # reshape for TensorFlow
    volume = np.expand_dims(volume, axis=0)   # batch
    volume = np.expand_dims(volume, axis=-1)  # channel

    prediction = model.predict(volume)

    mask = prediction[0]

    return mask
