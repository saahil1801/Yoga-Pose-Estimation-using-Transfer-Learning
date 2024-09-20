import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import tensorflow as tf
from model_utils import create_model

# Add the project root directory to the system path


def test_model_compilation():
    base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))
    model = create_model(base_model, 3)

    # Check if the model is compiled correctly with the expected loss function and optimizer
    assert model.loss == 'categorical_crossentropy'
    assert model.optimizer.learning_rate == 0.001
