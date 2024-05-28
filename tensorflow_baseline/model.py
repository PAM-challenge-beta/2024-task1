import tensorflow as tf

def load_model(model_path):
    """
    Loads a pre-trained CNN model from a file.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        tf.keras.Model: Loaded CNN model.
    """
    return tf.keras.models.load_model(model_path)

def create_model(input_shape, num_classes=2):
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