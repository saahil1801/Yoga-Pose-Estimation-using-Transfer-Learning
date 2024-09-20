import mlflow
import mlflow.tensorflow
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from data_utils import load_dataset, split_dataset
from model_utils import create_model
from config import Config
import os

# Load config from YAML file
config = Config()

def train_model(model, train_data, val_data, model_name, num_epochs):
    """
    Function to train the model and log results using MLflow.
    It saves the best model checkpoint and records the training history.
    
    Parameters:
    - model: Keras model to train
    - train_data: Training dataset
    - val_data: Validation dataset
    - model_name: Name of the model architecture being trained
    - num_epochs: Number of epochs to train
    """
    mlflow.set_experiment("Yoga Pose Classification")
    mlflow.tensorflow.autolog()

    checkpoint_path = f"models/best_model_{model_name}.keras"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    print(f"[INFO] Training model: {model_name}")
    
    # Start training
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=num_epochs,
        callbacks=[checkpoint]
    )

    # Plot training history
    plot_training_history(history, model_name)

    return checkpoint_path, history

def plot_training_history(history, model_name):
    """
    Function to plot training and validation loss and accuracy, and save the plots to files.
    
    Parameters:
    - history: The training history object returned by model.fit()
    - model_name: Name of the model architecture for labeling the plots
    """
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    # Plot accuracy
    if 'categorical_accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.legend()
    else:
        print(f"[WARNING] Categorical Accuracy not found in history for {model_name}")

    # Save the plot to a file
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(f"{output_dir}/{model_name}_training_history.png")
    plt.close()  # Close the figure to free up memory

def main():
    # Get configurations
    paths = config.get_paths()
    dataset_params = config.get_dataset_params()
    training_params = config.get_training_params()

    # Split dataset and prepare data loaders
    split_dataset(paths)
    train_dataset = load_dataset(paths['train_dir'], dataset_params)
    val_dataset = load_dataset(paths['val_dir'], dataset_params)

    models = [
        ("VGG16", tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))),
        ("ResNet50", tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))),
        ("InceptionV3", tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))),
        ("MobileNetV2", tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3)))
    ]

    best_model_name = None
    best_val_loss = float('inf')
    best_model_weights_path = None

    # Train each model and track the best one
    for model_name, base_model in models:
        model = create_model(base_model, len(dataset_params['class_names']))
        checkpoint_path, history = train_model(model, train_dataset, val_dataset, model_name, training_params['num_epochs'])

        # Check if this model has the best validation loss
        final_val_loss = history.history['val_loss'][-1]
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_model_name = model_name
            best_model_weights_path = checkpoint_path

    # Save the best model as "best_overall_model.keras"
    if best_model_name:
        print(f"[INFO] Best model is {best_model_name} with validation loss: {best_val_loss}")
        final_best_model_path = "models/best_overall_model.keras"
        shutil.copy(best_model_weights_path, final_best_model_path)
        mlflow.log_artifact(final_best_model_path)
        print(f"[INFO] Best model saved as: {final_best_model_path}")

if __name__ == "__main__":
    main()
