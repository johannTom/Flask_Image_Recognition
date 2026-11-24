"""Load the hand-sign model and prepare images for prediction."""

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """Convert an image to RGB, resize it, and normalize its pixels."""
    with Image.open(img_path) as image:
        img_resize = image.convert("RGB").resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(predict):
    """Return the class index with the highest prediction score."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
