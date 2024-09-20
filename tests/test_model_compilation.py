import unittest
import unittest
import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from model_utils import create_model
import tensorflow as tf

class TestModelCompilation(unittest.TestCase):

    def test_model_compilation(self):
        base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))
        model = create_model(base_model, 3)

        # Check if the model is compiled correctly with the expected loss function and optimizer
        self.assertEqual(model.loss, 'categorical_crossentropy')
        self.assertEqual(model.optimizer.learning_rate, 0.001)

if __name__ == '__main__':
    unittest.main()
