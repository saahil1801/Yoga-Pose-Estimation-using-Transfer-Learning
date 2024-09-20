import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from model_utils import create_model
from config import Config
import pytest
# Add the project root directory to the system path

@pytest.fixture
def config():
    return Config()


def test_model_accuracy():
    # Load the best model and validate its performance
    paths = config.get_paths()
    dataset_params = config.get_dataset_params()
    
    model = tf.keras.models.load_model("models/best_model_VGG16.keras")

    base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(256, 256, 3))
    model = create_model(base_model, num_classes=len(dataset_params['class_names']))

    # Load the checkpoint weights
    model.load_weights("models/checkpoint_model.h5")


    # Simulate or use actual validation data
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        'data/val',
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(256, 256),
        shuffle=False
    )

    loss, acc, top_k_acc = model.evaluate(val_dataset)

    # Check if the accuracy meets the minimum acceptance criteria
    assert acc >= 0.85  # e.g., the model should achieve >= 85% accuracy
