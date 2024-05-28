import pandas as pd
import librosa
import math
import numpy as np
import json
from dev_utils.constants import IMG_HEIGHT, IMG_WIDTH
from pathlib import Path
from dev_utils.preprocessing import preprocess_audio_segment, load_data
from tqdm import tqdm

import tensorflow_baseline.model as tf_model
import pytorch_baseline.model as torch_model

def run_cnn(data_dir, model_path, audio_representation, output_csv="detections.csv", threshold=0.5, deep_learning_library="tensorflow", device="cpu"):
    """
    Processes audio files to detect events using a CNN model.

    Args:
        data_dir (str): Directory containing audio files.
        model_path (str): Path to the trained TensorFlow model.
        audio_representation (str): Path to the JSON file containing configuration for spectrogram.
        output_csv (str): Path to save the detections CSV file.
        threshold (float): Probability threshold for considering a detection.
    """
    data_dir = Path(data_dir)

    # Load the audio representation settings, in this case, it is the configuration for our spectrogram
    with open(audio_representation, 'r') as file:
        config = json.load(file)

    audio_files = [audio_file for audio_file in data_dir.rglob('*.wav')]

    # Load model
    if deep_learning_library == "tensorflow":
        model = tf_model.load_model(model_path, device=device)
        
    elif deep_learning_library == "pytorch":
        model = torch_model.load_model(model_path, device=device)
        model.eval()

    output_csv_path = Path(output_csv).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=['filename', 'timestamp']).to_csv(output_csv_path, index=False)

    for file in tqdm(audio_files, desc='Processing Files'):
        file_total_duration = librosa.get_duration(path=file)
        num_segments = math.ceil(file_total_duration / config['duration'])

        detections = [] # List of detections for the current file
        for i in range(num_segments):
            start = i * config['duration']
            end = min((i + 1) * config['duration'], file_total_duration)
            y, sr = load_data(path=file, start=start, end=end, new_sr=config['sr'])

            # Check if the last segment needs to be padded or dropped
            if len(y) < config['duration'] * config['sr']:
                padding_length = int(config['duration'] * config['sr']) - len(y)
                # Padding the last segment with its reflection
                y = np.pad(y, (0, padding_length), mode='reflect')

            audio_representation = preprocess_audio_segment(y, sr, window_size=config['window'], step_size=config['step'])
            #audio_representation = audio_representation.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1) # I removed this line, @Bruno check the tensorflow version

            # Predict the segment$
            predictions = model.predict(audio_representation)

            
            # Check for positive class with threshold
            if predictions[0][1] > threshold:  # Index [0][1] for class 1 probability
                midpoint = (start + end) / 2
                detections.append({'filename': str(file.name), 'timestamp': midpoint})

        # Save detections to CSV
        if detections:
            df = pd.DataFrame(detections)
            df.to_csv(output_csv_path, index=False, mode='a', header=False)
    print("Detections saved to:", str(output_csv_path))

def main():
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the audio files')
    parser.add_argument('model_path', type=str, help='Path to the directory containing the saved model')
    parser.add_argument('audio_representation', type=str, help='Path to the audio representation config file .json')
    parser.add_argument('--output_csv', default='detections.csv', type=str, help='Path to save the detections CSV file.')
    parser.add_argument('--threshold', default=0.5, type=float, help="Detection threshold. Default is 0.5.")
    parser.add_argument('--deep_learning_library', default="tensorflow", type=str, help='The deep learning library to use (either pytorch or tensorflow)')
    parser.add_argument('--device', default="cpu", type=str, help='Device to run the code')
    args = parser.parse_args()

    run_cnn(**vars(args))

if __name__ == "__main__":
    main()