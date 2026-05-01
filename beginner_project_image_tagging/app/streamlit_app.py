import streamlit as st
import sys, os

st.write("Step 1: App started")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.write("Step 2: Path fixed")

from PIL import Image
st.write("Step 3: PIL loaded")

from src.predict import predict_image
st.write("Step 4: predict imported")

file = st.file_uploader("Upload Image")

if file:
    st.write("Step 5: File uploaded")

    image = Image.open(file)
    st.image(image)

    st.write("Step 6: Before prediction")

    result = predict_image(image)

    st.write("Step 7: After prediction")
    st.success(f"Prediction: {result}")