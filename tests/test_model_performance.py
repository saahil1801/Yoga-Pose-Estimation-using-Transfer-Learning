import tensorflow as tf
import sys
import os
import pytest

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from config import Config

@pytest.fixture
def config():
    return Config()

def test_model_accuracy(config):
    # Path to the saved best model
    model_path = "models/best_overall_model.keras"
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        pytest.fail(f"Model file not found: {model_path}")
    
    # Load the entire model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
    
    # Load validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        'data/val',  # Use config to get the validation directory
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(256, 256),
        shuffle=False
    )

    # Evaluate the model
    try:
        loss, acc, top_k_acc = model.evaluate(val_dataset)
    except Exception as e:
        pytest.fail(f"Model evaluation failed: {e}")

    # Check if the accuracy meets the minimum acceptance criteria
    assert acc >= 0.85, f"Model accuracy {acc} is below the required threshold of 80%."
