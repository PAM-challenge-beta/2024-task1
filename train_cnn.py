from pathlib import Path
from dev_utils.constants import IMG_HEIGHT, IMG_WIDTH

from pytorch_baseline import training_routine as torch_training_routine
from tensorflow_baseline import training_routine as tf_training_routine

def train_cnn(hdf5_db, train_table="/train", val_table=None, epochs=10, batch_size=32, output_folder=None, seed=None, deep_learning_library="tensorflow", device="cpu"):
    """
    Trains a CNN model on the provided dataset.
    It supposes that the dataset is stored in an HDF5 file.
    See create_db.py for more information on how to create the dataset.
    
    Args:
        hdf5_db (str): Path to the HDF5 database file.
        train_table (str): Path to the training data table within the HDF5 file.
        val_table (str): Path to the validation data table within the HDF5 file.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per batch.
        output_folder (str): Directory to save the trained model.
        seed (int): Seed for random number generator.
        deep_learning_library (str): The deep learning library to use (either pytorch or tensorflow).
        
    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    
    if deep_learning_library == "tensorflow":
        trained_model, history = tf_training_routine.train_model(hdf5_db, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=2,
                                                                 train_table=train_table, val_table=val_table, epochs=epochs, batch_size=batch_size, seed=seed, device=device)
    
    elif deep_learning_library == "pytorch":
        trained_model, history = torch_training_routine.train_model(hdf5_db, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=2, 
                                                                    train_table=train_table, val_table=val_table, epochs=epochs, batch_size=batch_size, seed=seed, device=device)

    else:
        raise ValueError(f"Invalid deep learning library: {deep_learning_library}, only PyTorch and TensorFlow are supported.")
    
    # Save the trained model to the output directory
    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    
    trained_model.save(output_folder)
    print(f"Model saved to {str(output_folder)}")

    return trained_model, history


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
    parser.add_argument('--deep_learning_library', default="tensorflow", type=str, help='The deep learning library to use (either pytorch or tensorflow)')
    parser.add_argument('--device', default="cpu", type=str, help='Device to run the code')


    # parser.add_argument('--checkpoints', default=None, type=int, help='Checkpoint frequency in terms of epochs.')
    args = parser.parse_args()

    model, history = train_cnn(**vars(args))
    if args.deep_learning_library == "tensorflow":
        tf_training_routine.plot_training_history(history)

    elif args.deep_learning_library == "pytorch":
        torch_training_routine.plot_losses(history)

if __name__ == "__main__":
    main()