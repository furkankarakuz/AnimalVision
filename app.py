from tensorflow.keras.preprocessing import image as img_proc
from PIL import Image
from model_process import load_model, predict_image, json_data
import streamlit as st
import warnings
import altair as alt
import pandas as pd


warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="AnimalVision", page_icon="🐾")

st.title("🐾 AnimalVision 🐾")
st.divider()
st.markdown("""<style>section[data-testid="stSidebar"] {width: 350px !important;}</style>""", unsafe_allow_html=True)

model, class_names = load_model()

file_upload = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns(2)


def control():
    if file_upload:
        with col1:
            img = img_proc.load_img(file_upload, target_size=(224, 224))
            st.image(Image.open(file_upload).resize((650, 650)))
            predicted_class, confidence = predict_image(img, img_proc, model, class_names)
        with col2:
            image_data, result_predict, result_confidence = json_data(predicted_class, confidence)
            st.markdown(f"<center><h4>Animal:  {result_predict.title()}</h4></center>", unsafe_allow_html=True)
            st.markdown(f"<center><h4>Predict Score : {result_confidence}</h4></center>", unsafe_allow_html=True)
            st.divider()
            bar_chart = (alt.Chart(pd.DataFrame(image_data)).mark_bar().encode(
                x=alt.X("confidence:Q", title="Confidence Score"),
                y=alt.Y("predicted_class:N", title="Predicted Class", sort='-x'),
                color=alt.Color("confidence:Q", scale=alt.Scale(scheme="blues"), legend=None), tooltip=["predicted_class", "confidence"])
                .properties(
                    width=500,
                    height=500,
                    title="Top 5 Predicted Classes"
            )
            )
            st.altair_chart(bar_chart, use_container_width=True, theme=None)


predict_button = st.sidebar.button("Predict!", on_click=control)
