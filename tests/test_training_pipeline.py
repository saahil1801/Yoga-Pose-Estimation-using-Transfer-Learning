import sys
import os
import pytest
import tensorflow as tf
import mlflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from train import main
from model_utils import create_model
from data_utils import load_dataset
from config import Config

# Add the project root directory to the system path

@pytest.fixture
def config():
    return Config()

def test_train_model_pipeline(config):
    paths = config.get_paths()
    dataset_params = config.get_dataset_params()
    
    train_dataset = load_dataset(paths['train_dir'], dataset_params)
    val_dataset = load_dataset(paths['val_dir'], dataset_params)

    base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(256, 256, 3))
    model = create_model(base_model, len(dataset_params['class_names']))
    
    # Simulate a training step
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)

    # Assert that training completed
    assert len(history.history['loss']) > 0

def test_pipeline_creates_model_artifacts():
    main()

    # Assert that the model artifact is created
    assert os.path.exists("models/best_overall_model.keras")
