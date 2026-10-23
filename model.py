# Importing required libs
"""Image preprocessing and prediction using the trained Keras model."""

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

# Load the pretrained digit recognition model.
model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """Resize and normalize an uploaded image for model prediction."""
    image = Image.open(img_path)
    resized_image = image.resize((224, 224))
    image_array = img_to_array(resized_image) / 255.0

    return image_array.reshape(1, 224, 224, 3)


def predict_result(predict):
    """Return the predicted digit with the highest model confidence."""
    predictions = model.predict(predict)
    return np.argmax(predictions[0], axis=-1)
