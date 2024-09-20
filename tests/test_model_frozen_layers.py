import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import tensorflow as tf
from model_utils import create_model



def test_frozen_layers():
    base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(256, 256, 3))
    model = create_model(base_model, 5)

    # Ensure all base model layers are frozen (not trainable)
    for layer in base_model.layers:
        assert not layer.trainable
