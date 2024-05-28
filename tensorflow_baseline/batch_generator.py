import tables as tb
import numpy as np
import tensorflow as tf

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
