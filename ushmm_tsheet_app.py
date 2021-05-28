import streamlit as st
import pandas as pd
import os
import urllib
from PIL import ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from PIL import Image
from google_drive_downloader import GoogleDriveDownloader as gdd


def main():
    st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 1500px;
        }}
    </style>
    """,
            unsafe_allow_html=True,
        )
    st.sidebar.image("images/si_logo.png")
    st.title('Target Sheet Classifier')
    st.sidebar.write("This app is run via Streamlit. It is designed for those at the United States Holocaust Memorial Museum. It hosts a binary classification modelwhich is meant to identify target sheets in microfilms. Special thanks to Mike Trizna at the Smithsonian Institution's Data Science Lab for helping me get this app in working order. The model's prediction will determine the likelyhood that something is a target sheet.")
    #download the model
    download_model()

    #load the model
    model = load_model()

    #The BASE model is Inception, which expects (299, 299, 3), so we set the size of the input image.
    image_size = (299, 299)

    st.sidebar.markdown("Upload an image to determine if it is a target sheet.")
    image = st.sidebar.file_uploader("", IMAGE_TYPES)

    #Once the user loads the image, we process it via the the model
    if image:
        image_data = image.read()
        with open("temp_image.jpg","wb") as f:
            f.write(image_data)
        image_size = (299,299)

        st.image(image_data)
        img = keras.preprocessing.image.load_img("temp_image.jpg", target_size=image_size)

        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0][0]
        inv_score = 1-score
        percentage = "{:.2%}".format(score)
        inv_percentage = "{:.2%}".format(inv_score)

        st.write(score)
        if score > .90:
            st.write(f"This is likely a Target Sheet. The model has a {percentage} confidence.")
        else:
            st.write(f"This is likely NOT a Target Sheet. The model has a {inv_percentage} confidence.")

def download_model():
    pb_file = "1FLKcLJ_KmvQy7htItH4m7ACPIicBBZaK"
    meta_file = "1wT4K_9pHQbPuQS9iKnxTU0LzBehAyEs6"
    index_file = "1NzRq2M_fEY0in7YUtD71ggGrgPNgSe8S"
    weights_file = "1EUaakEthwSnIY8dyj9qAbu5sCekJ0g59"

    if os.path.exists("model"):
        pass
    else:
        os.mkdir("model")
    if os.path.exists("model/variables"):
        pass
    else:
        os.mkdir("model/variables")

    #download the pb
    if os.path.exists("model/saved_model.pb"):
        pass
    else:
        gdd.download_file_from_google_drive(file_id=f'{pb_file}',
                                    dest_path='./model/saved_model.pb')
    #download keras metadata
    if os.path.exists("model/keras_metadata.pb"):
        pass
    else:
        gdd.download_file_from_google_drive(file_id=f'{meta_file}',
                                    dest_path='./model/keras_metadata.pb')
    #download keras index
    if os.path.exists("model/variables/varibales.index"):
        pass
    else:
        gdd.download_file_from_google_drive(file_id=f'{index_file}',
                                    dest_path='./model/variables/variables.index')
    #download keras weights
    if os.path.exists("model/variables/variables.data-00000-of-00001"):
        pass

    else:
        gdd.download_file_from_google_drive(file_id=f'{weights_file}',
                                    dest_path='./model/variables/variables.data-00000-of-00001')
IMAGE_TYPES = ["png", "jpg"]
#Cache the model in memory
@st.cache(allow_output_mutation=True)
def load_model():
    inf_model = keras.models.load_model("model")
    return inf_model


if __name__ == "__main__":
    main()
