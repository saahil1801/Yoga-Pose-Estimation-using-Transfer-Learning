import unittest
import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from train import main
from model_utils import create_model
from data_utils import load_dataset
from config import Config
import os
import tensorflow as tf
import mlflow 

class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        self.config = Config()

    def test_train_model_pipeline(self):
        paths = self.config.get_paths()
        dataset_params = self.config.get_dataset_params()
        train_dataset = load_dataset(paths['train_dir'], dataset_params)
        val_dataset = load_dataset(paths['val_dir'], dataset_params)

        base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(256, 256, 3))
        model = create_model(base_model, len(dataset_params['class_names']))

        # Simulate a training step
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)

        # Assert that training completed
        self.assertGreater(len(history.history['loss']), 0)

    def test_pipeline_creates_model_artifacts(self):
        main()

        # Assert that model artifact is created
        self.assertTrue(os.path.exists("models/best_overall_model.keras"))



if __name__ == '__main__':
    unittest.main()
