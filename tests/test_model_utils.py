import unittest
import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from data_utils import load_dataset, split_dataset
from config import Config
from pathlib import Path

class TestDataUtils(unittest.TestCase):

    def setUp(self):
        self.config = Config()

    def test_load_dataset(self):
        dataset_params = self.config.get_dataset_params()
        dataset = load_dataset(self.config.get_paths()['train_dir'], dataset_params)
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset_params['class_names']), 5)

    def test_split_dataset(self):
        paths = self.config.get_paths()
        split_dataset(paths)
        # Test if the split directories are created
        self.assertTrue(Path(paths['train_dir']).exists())
        self.assertTrue(Path(paths['val_dir']).exists())
        self.assertTrue(Path(paths['test_dir']).exists())

if __name__ == '__main__':
    unittest.main()
