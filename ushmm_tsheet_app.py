import streamlit as st
import altair as alt
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
    st.title('Target Sheet Classifier')

    download_model()

    model = load_model()
    image_size = (299, 299)

    st.markdown("Upload an image to determine if it is a target sheet.")
    image = st.file_uploader("", IMAGE_TYPES)
    if image:
        image_data = image.read()
        with open("temp_image.jpg","wb") as f:
            f.write(image_data)
        image_size = (299,299)

        st.image(image_data, use_column_width=True)
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
    weights = "1EUaakEthwSnIY8dyj9qAbu5sCekJ0g59"

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
        gdd.download_file_from_google_drive(file_id='1FLKcLJ_KmvQy7htItH4m7ACPIicBBZaK',
                                    dest_path='./model/saved_model.pb')

    #download keras metadata
    if os.path.exists("model/keras_metadata.pb"):
        pass
    else:
        gdd.download_file_from_google_drive(file_id=f'{meta_file}',
                                    dest_path='./model/keras_metadata.pb')
    #download keras metadata
    if os.path.exists("model/variables/varibales.index"):
        pass
    else:
        gdd.download_file_from_google_drive(file_id='1NzRq2M_fEY0in7YUtD71ggGrgPNgSe8S',
                                    dest_path='./model/variables/variables.index')


    #download keras weights
    if os.path.exists("model/variables/variables.data-00000-of-00001"):
        pass

    else:
        gdd.download_file_from_google_drive(file_id='1EUaakEthwSnIY8dyj9qAbu5sCekJ0g59',
                                    dest_path='./model/variables/variables.data-00000-of-00001')


def predictions_to_chart(prediction, classes):
    img = keras.preprocessing.image.load_img(file, target_size=image_size)

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]

    pred_rows = []
    for i, conf in enumerate(list(prediction[2])):
        pred_row = {'class': classes[i],
                    'probability': round(float(conf) * 100,2)}
        pred_rows.append(pred_row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.head()
    top_probs = pred_df.sort_values('probability', ascending=False).head(4)
    chart = (
        alt.Chart(top_probs)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=(0, 100))),
            y=alt.Y("class:N",
                    sort=alt.EncodingSortField(field="probability", order="descending"))
        )
    )
    return chart

@st.cache(allow_output_mutation=True)
def load_model():
    inf_model = keras.models.load_model("model")
    return inf_model

#
# def download_file(file_path):
#     # Don't download the file twice. (If possible, verify the download using the file length.)
#     if os.path.exists(file_path):
#         if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
#             return
#         elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
#             return
#
#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading %s..." % file_path)
#         progress_bar = st.progress(0)
#         with open(file_path, "wb") as output_file:
#             with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
#                 length = int(response.info()["Content-Length"])
#                 counter = 0.0
#                 MEGABYTES = 2.0 ** 20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)
#
#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
#                         (file_path, counter / MEGABYTES, length / MEGABYTES))
#                     progress_bar.progress(min(counter / length, 1.0))
#
#     # Finally, we remove these visual elements by calling .empty().
#     finally:
#         if weights_warning is not None:
#             weights_warning.empty()
#         if progress_bar is not None:
#             progress_bar.empty()
#
#     return

IMAGE_TYPES = ["png", "jpg"]

EXTERNAL_DEPENDENCIES = {
    "pet_classifier_resnet34.pkl": {
        "url": "https://www.dropbox.com/s/feox6arze2gc14q/pet_classifier_resnet34.pkl?dl=1",
        "size": 87826330
    }
}

if __name__ == "__main__":
    main()
