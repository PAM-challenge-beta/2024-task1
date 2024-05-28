import tensorflow as tf

from matplotlib import pyplot as plt
from pathlib import Path

from model import create_model
from batch_generator import BatchGenerator  

def train_model(hdf5_db, input_shape=(128,128,1), num_classes=2, train_table="/train", val_table=None, epochs=10, batch_size=32, seed=None):
    """
    Trains a CNN model on the provided dataset.
    
    Args:
        hdf5_db (str): Path to the HDF5 database file.
        input_shape (tuple): Shape of the input images.
        train_table (str): Path to the training data table within the HDF5 file.
        val_table (str): Path to the validation data table within the HDF5 file.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per batch.
        output_folder (str): Directory to save the trained model.
        seed (int): Seed for random number generator.
        
    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    if seed:
        # Set the random seed for reproducibility
        tf.random.set_seed(seed)
    
    model = create_model(input_shape, num_classes)

    # Compile the model with optimizer, loss, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Create the training data generator
    train_batch_generator = BatchGenerator(hdf5_db, train_table, batch_size, num_classes)
    
    val_generator = None
    if val_table:
        # Create the validation data generator if validation table exists
        val_generator = BatchGenerator(hdf5_db, val_table, batch_size, num_classes)

    # Train the model
    history = model.fit(train_batch_generator, epochs=epochs, validation_data=val_generator)

    return model, history

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.
    
    Args:
        history (tf.keras.callbacks.History): Training history object.
    """

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['categorical_accuracy'])
    if 'val_categorical_accuracy' in history.history:
        plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

# def main():
#     import argparse

#     # parse command-line args
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('hdf5_db', type=str, help='HDF5 Database file path')
#     parser.add_argument('--train_table', default='/train', type=str, help="The table within the hdf5 database where the training data is stored. For example, /train")
#     parser.add_argument('--val_table', default=None, type=str, help="The table within the hdf5 database where the training data is stored. For example, /val")
#     parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
#     parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
#     parser.add_argument('--output_folder', default=None, type=str, help='Output directory')
#     parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
#     parser.add_argument('--depp_learning_library', default="tensorflow", type=str, help='The deep learning library to use (either pytorch or tensorflow)')

#     # parser.add_argument('--checkpoints', default=None, type=int, help='Checkpoint frequency in terms of epochs.')
#     args = parser.parse_args()

#     history = train_model(**vars(args))
#     plot_training_history(history)

# if __name__ == "__main__":
#     main()