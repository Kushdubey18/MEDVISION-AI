import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


# Load Model
model = tf.keras.models.load_model(
    "model/pneumonia_model_v2.keras"
)


def predict_image(image_path):

    # ==========================
    # Load Image
    # ==========================

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # ==========================
    # Prediction
    # ==========================

    prediction = model.predict(img_array)

    probability = float(prediction[0][0])


    # ==========================
    # Disease Result
    # ==========================

    if probability > 0.5:

        disease = "PNEUMONIA"
        confidence = probability * 100

    else:

        disease = "NORMAL"
        confidence = (1 - probability) * 100


    # ==========================
    # Create Heatmap
    # ==========================

    heatmap_path = "static/heatmap.png"


    image = cv2.imread(image_path)


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    heatmap = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET
    )


    cv2.imwrite(
        heatmap_path,
        heatmap
    )


    # ==========================
    # Return Output
    # ==========================

    return disease, confidence, heatmap_path