import tensorflow as tf
import tables as tb
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from dev_utils.constants import IMG_HEIGHT, IMG_WIDTH


class BatchGenerator(tf.keras.utils.Sequence):
    """
    Custom Generator for generating batches of data from an HDF5 file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file containing the data.
        table_path (str): Path to the table within the HDF5 file.
        batch_size (int): Number of samples per batch.
        num_classes (int): Number of classes for one-hot encoding the labels.
    """
    def __init__(self, hdf5_path, table_path, batch_size, num_classes):
        self.hdf5_path = hdf5_path
        self.table_path = table_path + "/data"
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.indices = None
        self.on_epoch_end()

    def __len__(self):
        with tb.open_file(self.hdf5_path, 'r') as file:
            table = file.get_node(self.table_path)
            return int(np.ceil(table.nrows / self.batch_size))

    def __getitem__(self, idx):
        """
        Generates one batch of data.
        
        Args:
            idx (int): Index of the batch.
        
        Returns:
            tuple: Batch of data and corresponding labels.
        """
        # Start and end indices for the batch
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices)) # Ensure we do not go out of bounds in the last batch if it is smaller than batch_size
        batch_indices = self.indices[start_idx:end_idx]

        # Load data from the HDF5 file
        with tb.open_file(self.hdf5_path, 'r') as file:
            table = file.get_node(self.table_path)
            # Extract data and labels for the batch
            X = np.array([table[row]['data'] for row in batch_indices])
            y = np.array([table[row]['label'] for row in batch_indices])

            # Reshape data to match the input shape required by the model
            # Convert to tensor Batch_size, Height, Width, channels
            X = tf.reshape(X, (X.shape[0],X.shape[1], X.shape[2],1))
            # Convert labels to one-hot encoding
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
            return X, y

    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        with tb.open_file(self.hdf5_path, 'r') as file:
            table = file.get_node(self.table_path)
            self.indices = np.arange(table.nrows)
            np.random.shuffle(self.indices)

def create_model(num_classes=2, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """
    Creates a CNN model for image classification.
    
    Args:
        num_classes (int): Number of classes for the output layer.
        input_shape (tuple): Shape of the input images.
        
    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = tf.keras.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(256, (3, 3), input_shape=input_shape),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(128, (3, 3)),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.Activation('relu'),                       
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten the feature maps
        tf.keras.layers.Flatten(),
        # Fully connected layer
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.Activation('relu'),

        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_cnn(hdf5_db, train_table="/train", val_table=None, epochs=10, batch_size=32, output_folder=None, seed=None):
    """
    Trains a CNN model on the provided dataset.
    
    Args:
        hdf5_db (str): Path to the HDF5 database file.
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
    
    model = create_model()

    # Compile the model with optimizer, loss, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Create the training data generator
    train_generator = BatchGenerator(hdf5_db, train_table, batch_size, 2)
    
    val_generator = None
    if val_table:
        # Create the validation data generator if validation table exists
        val_generator = BatchGenerator(hdf5_db, val_table, batch_size, 2)

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.parent.mkdir(parents=True, exist_ok=True)

    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
    # Save the trained model to the output directory
    model.save(output_folder)
    print(f"Model saved to {str(output_folder)}")

    return history

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


def main():
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('hdf5_db', type=str, help='HDF5 Database file path')
    parser.add_argument('--train_table', default='/train', type=str, help="The table within the hdf5 database where the training data is stored. For example, /train")
    parser.add_argument('--val_table', default=None, type=str, help="The table within the hdf5 database where the training data is stored. For example, /val")
    parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--output_folder', default=None, type=str, help='Output directory')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')

    # parser.add_argument('--checkpoints', default=None, type=int, help='Checkpoint frequency in terms of epochs.')
    args = parser.parse_args()

    history = train_cnn(**vars(args))
    plot_training_history(history)

if __name__ == "__main__":
    main()