import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def save_gradcam(img_path,
                 model,
                 last_conv_layer_name,
                 save_path):

    img = tf.keras.utils.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name
    )

    img = cv2.imread(img_path)

    img = cv2.resize(
        img,
        (224, 224)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    jet = cm.get_cmap("jet")

    jet_colors = jet(
        np.arange(256)
    )[:, :3]

    jet_heatmap = jet_colors[
        heatmap
    ]

    jet_heatmap = cv2.resize(
        jet_heatmap,
        (img.shape[1], img.shape[0])
    )

    jet_heatmap = np.uint8(
        255 * jet_heatmap
    )

    superimposed_img = cv2.addWeighted(
        img,
        0.6,
        jet_heatmap,
        0.4,
        0
    )

    cv2.imwrite(
        save_path,
        superimposed_img
    )

    return save_path
