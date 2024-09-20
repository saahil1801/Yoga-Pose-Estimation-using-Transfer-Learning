import sys
import os
# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import tensorflow as tf
from model_utils import create_model



def test_create_model_output_shape():
    base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(256, 256, 3))
    model = create_model(base_model, 5)
    assert model.output_shape == (None, 5)  # Check output shape matches class count

def test_create_model_different_input_shape():
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = create_model(base_model, 10)
    assert model.output_shape == (None, 10)  # Check output shape for 10 classes
    assert model.input_shape == (None, 224, 224, 3)  # Ensure input shape is preserved
