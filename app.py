import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import mlflow.pyfunc

# Load the ResNet model
model_name = "resent"
alias = "challenger"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")


# Class names
class_name = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

# Streamlit app
st.title("Yoga Pose Classification")

uploaded_file = st.file_uploader("Upload an image of a yoga pose", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
    # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_array = np.array(image)
        img_array = tf.image.resize(img_array, (256, 256))
        img_array = np.expand_dims(img_array, axis=0) 

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        st.write(f"Predicted Pose: {class_name[predicted_class]}")
        