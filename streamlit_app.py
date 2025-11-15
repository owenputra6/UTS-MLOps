import streamlit as st
from PIL import Image
import pandas as pd
from predict import predict_one, class_names

st.title("Mid-Test MLOps X-ray image dissease classification")
st.write(f"Upload an image, I’ll classify it as either {class_names}")

uploaded_file = st.file_uploader("Upload here (one image only)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded!", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Loading, sabar ya"):
            pred_class, confidence, probs = predict_one(img)

        st.subheader("Prediction result")
        st.write(f"Predicted class: {pred_class}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
