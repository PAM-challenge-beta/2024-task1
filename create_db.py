import numpy as np
import pandas as pd
import librosa
import json
import tables as tb
import soundfile as sf
from dev_utils.constants import IMG_HEIGHT, IMG_WIDTH
from tqdm import tqdm
from pathlib import Path
from dev_utils.preprocessing import define_intervals, create_random_segments, file_duration_table, load_data, preprocess_audio_segment
from dev_utils.hdf5_helper import SpectrogramTable, insert_spectrogram_data, create_or_get_table
from skimage.transform import resize


def create_db(data_dir, audio_representation, annotations=None, output="db.h5", 
              table_name='/train', num_background='same', overwrite=False, exclude_subdirs=None):

    # Load annotations
    annotations = pd.read_csv(annotations)

    # Load the audio representation settings, in this case, it is the configuration for our spectrogram
    with open(audio_representation, 'r') as file:
        config = json.load(file)

    # Our annotations have different lengths. Lets modify them and ensure they are all of the same duration. In this case, 3 seconds.
    annotations = define_intervals(annotations, duration=config['duration'], center=True)

    # Filter out annotations in excluded subdirectories
    if exclude_subdirs:
        annotations = annotations[~annotations['filename'].str.contains('|'.join(exclude_subdirs))]

    # Lets add a label to them to differentiate from background segments
    annotations['label'] = 1

    if num_background == 'same':
        num_background = len(annotations)

    print("Extracting random background segments...")
    files = file_duration_table(data_dir, exclude_subdirs=exclude_subdirs)
    background = create_random_segments(files, duration=config['duration'], num=num_background, annotations=annotations)
    background['label'] = 0

    # Concatenate annotations and background DataFrames to store in our hdf5 database
    combined_df = pd.concat([annotations, background], ignore_index=True)
    print('Creating db...')
    with tb.open_file(output, mode='w' if overwrite else 'a') as h5file:
        table = create_or_get_table(h5file, table_name, 'data', SpectrogramTable)
        print(f'Writing to table {table_name}.')
        for idx, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0]):
            y, sr = load_data(path=Path(data_dir) / row['filename'], start=row['start'], end=row['start'] + config['duration'], new_sr=config['sr'])

            audio_representation = preprocess_audio_segment(y, sr, window_size=config['window'], step_size=config['step'])

            insert_spectrogram_data(table, row['filename'], row['label'], audio_representation)
    

def main():
    import argparse

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    def num_background_type(value):
        if value.isdigit():
            return int(value)  # Convert to int if the value is digit-only
        elif value == 'same':
            return value  # Return the string 'same'
        else:
            raise argparse.ArgumentTypeError("Number of background samples must be an integer or 'same'.")

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the audio files')
    parser.add_argument('audio_representation', type=str, help='Path to the audio representation config file .json')
    parser.add_argument('annotations', type=str, help='Path to the annotations .csv')
    parser.add_argument('--table_name', default='/train', type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    parser.add_argument('--num_background', default='same', type=num_background_type, help="Number of backgorund samples to ectract by randomly sampling from all audio files while avoinding the annotations. Can be either an integer or 'same', in which case will match the number of annotations")
    parser.add_argument('--output', default='db.h5', type=str, help='HDF5 dabase name. For instance: db.h5')
    parser.add_argument('--exclude_subdirs', default=None,  nargs='+', type=str, help='Subdirs to exclude from the annotations. Usefull for splitting into different sets')
    parser.add_argument('--overwrite', default=False, type=boolean_string, help='Overwrite the database. Otherwise append to it.')
    args = parser.parse_args()

    create_db(**vars(args))

if __name__ == "__main__":
    main()