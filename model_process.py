import tensorflow as tf
import streamlit as st
import numpy as np


@st.cache_resource
def load_model():
    with open("AnimalList.txt", "r", encoding="utf-8") as file:
        content = file.read()
    animal_list = content.split("\n")
    return tf.keras.models.load_model("AnimalVisionModel.keras"), animal_list


def predict_image(img, img_proc, model, class_names):
    img_array = img_proc.img_to_array(img.resize((224, 224))) / 255
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    top_5_indices = np.argsort(predictions)[-5:][::-1]
    top_5_probs = [round(float(predictions[i]), 2) for i in top_5_indices]
    top5_class = [class_names[i] for i in top_5_indices]

    return top5_class, top_5_probs


def json_data(predicted_class, confidence):
    data = {}
    data["predicted_class"] = predicted_class
    data["confidence"] = confidence

    return data, predicted_class[0], confidence[0]
