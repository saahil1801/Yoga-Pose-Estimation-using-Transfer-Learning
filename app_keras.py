import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the best model
model = tf.keras.models.load_model("best_model_VGG16.keras")

# Class names
class_names = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

st.title("Yoga Pose Classification")

uploaded_file = st.file_uploader("Upload an image of a yoga pose", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = np.expand_dims(img_array, axis=0) 

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    st.write(f"Predicted Pose: {class_names[predicted_class]}")
    # st.write(f"Prediction probabilities: {predictions[0]}")
