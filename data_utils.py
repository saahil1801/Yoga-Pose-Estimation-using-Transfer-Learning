import tensorflow as tf
from pathlib import Path
import shutil
from imutils import paths
from sklearn.model_selection import train_test_split

def load_dataset(directory, dataset_params):
    """
    Load dataset from the specified directory using the parameters provided.

    Parameters:
    - directory: Path to the dataset directory (train/val/test).
    - dataset_params: Dictionary containing dataset-related configurations (class_names, image_size, batch_size, etc.)

    Returns:
    - A TensorFlow dataset.
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=dataset_params['class_names'],
        color_mode=dataset_params['color_mode'],
        batch_size=dataset_params['batch_size'],
        image_size=tuple(dataset_params['image_size']),  # Convert list to tuple
        shuffle=True,
        seed=dataset_params['seed']
    )

def split_image_folder(image_paths, folder):
    data_path = Path(folder)
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        full_path = Path(path)
        image_name = full_path.name
        label = full_path.parent.name
        label_folder = data_path / label
        if not label_folder.is_dir():
            label_folder.mkdir(parents=True, exist_ok=True)
        destination = label_folder / image_name
        shutil.copy(path, destination)

def split_dataset(config_paths):
    """
    Split the dataset into training, validation, and testing sets and create corresponding folders.
    
    Parameters:
    - config_paths: Dictionary containing paths for 'download_dir', 'train_dir', 'val_dir', and 'test_dir'
    """
    # Load all image paths from the download directory
    image_paths = list(sorted(paths.list_images(config_paths['download_dir'])))
    class_names = [Path(x).parent.name for x in image_paths]
    
    # Split into train and validation sets
    train_paths, rest_of_paths = train_test_split(image_paths, stratify=class_names, test_size=0.20, shuffle=True, random_state=42)
    
    # Further split the rest into validation and test sets
    val_paths, test_paths = train_test_split(rest_of_paths, stratify=[Path(x).parent.name for x in rest_of_paths], test_size=0.50, shuffle=True, random_state=42)
    
    # Create directories and split data into corresponding folders
    split_image_folder(train_paths, config_paths['train_dir'])
    split_image_folder(val_paths, config_paths['val_dir'])
    split_image_folder(test_paths, config_paths['test_dir'])
