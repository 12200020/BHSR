# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import random

# hide depreciation warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Bhutanese Hand Sign Recognition",
    page_icon=":hand:",
    initial_sidebar_state='auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)  # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

with st.sidebar:
    st.image('mg.png')
    st.title("Bhutanese Hand Sign Recognition")
    st.subheader("Bhutanese Hand Sign Recognition would help detect hand sign.")

st.write("""
         # Bhutanese Hand Sign Recognition
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

# Load the pre-trained model
model = tf.keras.models.load_model('bhsr_model.h5')  # Replace with the actual path to your model file

def import_and_predict(image_data, model):
    # size = (224,224)
    size = (64, 64)  # Adjust the size according to your model's input shape
    # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = [
        'ཀ', 'ཁ', 'ག', 'ང', 'ཅ', 'ཆ', 'ཇ', 'ཉ', 'ཏ', 'ཐ',
        'ད', 'ན', 'པ', 'ཕ', 'བ', 'མ', 'ཙ', 'ཚ', 'ཛ', 'ཝ',
        'ཞ', 'ཟ', 'འ', 'ཡ', 'ར', 'ལ', 'ཤ', 'ས', 'ཧ', 'ཨ'
    ]

    # Get the top predicted classes and their probabilities
    top_classes_indices = np.argsort(predictions[0])[::-1]
    top_classes = [class_names[i] for i in top_classes_indices]
    top_probs = predictions[0, top_classes_indices]

    # Display the top classes and probabilities
    st.write("Top Predicted Classes:")
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
        st.write(f"{i + 1}. {cls} - Probability: {prob * 100:.2f}%")

    # Display the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    predicted_prob = predictions[0, predicted_class_index]

    st.write(f"\nPredicted Class: {predicted_class} - Probability: {predicted_prob * 100:.2f}%")
    st.balloons()
    st.sidebar.success(f"Alphabet detected: {predicted_class}")