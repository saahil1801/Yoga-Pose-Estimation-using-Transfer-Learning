import sys
import os
from pathlib import Path
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from data_utils import load_dataset, split_dataset
from config import Config

# Add the project root directory to the system path


@pytest.fixture
def config():
    return Config()

def test_load_dataset(config):
    dataset_params = config.get_dataset_params()
    dataset = load_dataset(config.get_paths()['train_dir'], dataset_params)
    
    assert dataset is not None
    assert len(dataset_params['class_names']) == 5

def test_split_dataset(config):
    paths = config.get_paths()
    split_dataset(paths)

    # Test if the split directories are created
    assert Path(paths['train_dir']).exists()
    assert Path(paths['val_dir']).exists()
    assert Path(paths['test_dir']).exists()
