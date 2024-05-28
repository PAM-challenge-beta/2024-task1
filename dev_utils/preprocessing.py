import pandas as pd
import numpy as np
import os
import soundfile as sf
import librosa
from skimage.transform import resize
from dev_utils.constants import IMG_HEIGHT, IMG_WIDTH

def define_intervals(df, duration, center=True):
    """
    Modifies the dataframe to include new start and end times for intervals around annotated sections.

    For instance

    Parameters:
    - df (pd.DataFrame): DataFrame containing the annotations with columns 'start' and 'end'.
    - duration (float): Fixed duration of the new interval in seconds.
    - center (bool): If True, centers the new interval around the annotated section. 
                     If False, places the new interval randomly within the original annotated section.

    Returns:
    - pd.DataFrame: Modified DataFrame with updated 'start' and 'end' columns for the new intervals.
    """
    # Calculate original length of each annotation
    length = df['end'] - df['start']
    
    # Calculate new start times based on whether we are centering or placing randomly
    if center:
        # Center the new interval around the original midpoint
        df['start'] = df['start'] + 0.5 * (length - duration)
    else:
        # Randomly place the new interval within the original bounds
        df['start'] = df['start'] + np.random.rand(len(df)) * (length - duration)
    
    # Ensure that new start times are not negative
    df['start'] = np.maximum(df['start'], df['start'])
    
    # Calculate new end times, ensuring they do not extend beyond the original end time
    df['end'] = df['start'] + duration

    return df

def find_files(path, substr, search_subdirs=True, return_path=False, exclude_subdirs=None):
    """Search for files with specific substrings in their names, typically file extensions.

    Args:
        path (str): Base directory to search.
        substr (list): List of substrings to search for in file names.
        search_subdirs (bool): Whether to search subdirectories.
        return_path (bool): If True, path to each file, relative to the top directory. 
                            If false, only return the filenames 
        exclude_subdirs (list or None): List of subdirectory names to exclude from the search.

    Returns:
        list: A list of file paths or file names that match the criteria.
    """
    matched_files = []
    for root, dirs, files in os.walk(path):
        # Exclude specific subdirectories if any are specified
        if exclude_subdirs is not None:
            dirs[:] = [d for d in dirs if d not in exclude_subdirs]

        for file in files:
            if any(file.endswith(ext) for ext in substr):
                full_path = os.path.join(root, file)
                if return_path:
                    matched_files.append(os.path.relpath(full_path, start=path))
                else:
                    matched_files.append(file)
        if not search_subdirs:
            break
    return matched_files

def get_duration(file_paths):
    """Calculate the durations of multiple audio files.

    Args:
        file_paths (list): List of paths to audio files.

    Returns:
        list: Durations of the audio files in seconds.
    """
    durations = []
    for file_path in file_paths:
        with sf.SoundFile(file_path) as sound_file:
            durations.append(len(sound_file) / sound_file.samplerate)
    return durations

def file_duration_table(path, exclude_subdirs=None):
    """ Create file duration table.

        Args:
            path: str
                Path to folder with audio files (\*.wav)
            num: int
                Randomly sample a number of files
            exclude_subdir: str
                Exclude subdir from the search 

        Returns:
            df: pandas DataFrame
                File duration table. Columns: filename, duration, (datetime)
    """
    paths = find_files(path=path, substr=['.wav', '.WAV', '.flac', '.FLAC'], search_subdirs=True, return_path=True, exclude_subdirs=exclude_subdirs)
    durations = get_duration([os.path.join(path,p) for p in paths])
    df = pd.DataFrame({'filename':paths, 'duration':durations})

    return df

def validate_segment(annotations, fname, start, end, buffer):
    """
    Check if a segment overlaps with annotations or extends out of bounds with given buffer.
    
    Args:
        annotations (pd.DataFrame): DataFrame containing annotations with columns 'filename', 'start', and 'end'.
        fname (str): Name of the file for which the segment is being validated.
        start (float): Proposed start time of the segment.
        end (float): Proposed end time of the segment.
        buffer (float): Additional time buffer to apply around the segment for safety.

    Returns:
        bool: True if the segment does not overlap with any annotations (considering the buffer), False otherwise.
    """
    if annotations is not None:
        # Check for overlapping annotations, considering the buffer
        overlaps = annotations[
            (annotations['filename'] == fname) &
            ((annotations['start'] - buffer <= end) & (annotations['end'] + buffer >= start))
        ]
        if len(overlaps) > 0:
            return False  # Overlap found, segment is not valid
    return True  # No overlap, segment is valid


def create_random_segments(files, duration, num, annotations=None, buffer=0):
    """
    Generates a specified number of random audio segments from a list of files, ensuring no overlap with annotations.

    Args:
        files (pd.DataFrame): DataFrame with file information, must include 'filename' and 'duration'.
        duration (float): Length of each audio segment to generate in seconds.
        num (int): Number of audio segments to generate.
        annotations (pd.DataFrame, optional): DataFrame containing annotations to avoid, with columns 'filename', 'start', and 'end'.
        buffer (float): Buffer time in seconds to add around each segment to avoid annotations.

    Returns:
        pd.DataFrame: DataFrame containing generated segments with columns 'filename', 'start', and 'end'.

    Raises:
        Warning: If the number of generated samples is less than the requested number due to overlap or file duration limits.
    """    
    results = []
    attempts = 0
    max_attempts = num * 10  # Limit on attempts to prevent infinite loops

    # Filter out files that are too short to accommodate the segment plus buffer
    files = files[files['duration'] >= duration + 2 * buffer].reset_index(drop=True)
    # Calculate probability weights based on file duration for weighted sampling
    files['prob'] = files['duration'] / files['duration'].sum()

    while len(results) < num and attempts < max_attempts:
        # Sample a file based on duration probability
        chosen_file = files.sample(1, weights='prob', random_state=seed).iloc[0]
        # Calculate the maximum possible valid start time considering the duration and buffer
        max_start = chosen_file['duration'] - duration - buffer
        # Randomly determine the start time within the valid range
        start = buffer + np.random.uniform(0, max_start)
        end = start + duration

        # Validate the chosen segment against annotations
        if validate_segment(annotations, chosen_file['filename'], start, end, buffer):
            results.append({
                'filename': chosen_file['filename'],
                'start': start,
                'end': end
            })
        attempts += 1

    # Warn if the desired number of segments was not reached
    if len(results) < num:
        print(f"Warning: Only {len(results)} out of {num} requested samples could be generated.")

    return pd.DataFrame(results)

def load_data(path, start=None, end=None, new_sr=None):
    """ Loads a segment of an audio file.

    Args:
        path (str or Path): The path to the audio file.
        start (float): The start time in seconds from which to begin audio extraction. 
                            If None, extraction starts from the beginning of the file.
        end (float): The end time in seconds at which to stop audio extraction.
                            If None, extraction goes till the end of the file.
        new_sr (int): The new sample rate to which the audio should be resampled.
                            If None, the original sample rate is maintained.

    Returns:
        tuple: A tuple containing:
            - audio_segment (numpy.ndarray): The extracted audio segment.
            - return_sr (int): The sample rate of the returned audio segment.

    Raises:
        ValueError: If `start` is greater than `end` or if both are outside the duration of the audio.

    Notes:
        - If `start` is specified as negative, it adjusts `start=0` and `end` to maintain the intended duration segment.
        - If `end` exceeds the file's duration, it adjusts `end` to fit the intended duration within the file's length.
    """
    
    with sf.SoundFile(path) as file:
        sr = file.samplerate
        total_frames = file.frames
        file_duration = total_frames / sr  # Duration of the file in seconds

        # Default to the full file if neither start nor end is provided
        if start is None and end is None:
            start, end = 0, file_duration
        
        # Adjust start time if it's negative, and dynamically adjust the end time to maintain duration
        if start is not None and start < 0:
            end += -start  # Adjust end time by the amount start time was negative
            start = 0

        # Ensure end time does not exceed the file's duration
        if end is not None and end > file_duration:
            # start = max(start - (end - file_duration), 0)
            end = file_duration

        # Convert start and end times to frame indices
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        # Read the specific segment of the audio file
        file.seek(start_frame)
        audio_segment = file.read(end_frame - start_frame)

    return_sr = sr
    # Resample the audio segment if new sample rate is provided
    if new_sr is not None:
        audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=new_sr)
        return_sr = new_sr
    
    return audio_segment, return_sr

def preprocess_audio_segment(segment, sr, window_size, step_size):
    """
    Preprocesses an audio segment by converting it to a spectrogram using Short Time Fourier Transform (STFT),
    converting the amplitude to decibel units, resizing the spectrogram image, and normalizing it.

    Args:
        segment (numpy.ndarray): The audio segment as a 1D numpy array.
        sr (int): The sample rate of the audio segment.
        window_size (float): The window size for the STFT, given in seconds.
        step_size (float): The step size for the STFT, given in seconds.

    Returns:
        numpy.ndarray: The preprocessed spectrogram of the audio segment.

    """
    n_fft = int(window_size * sr)  # Window size for STFT
    hop_length = int(step_size * sr)  # Step size for STFT
    S = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length))
    spec = librosa.amplitude_to_db(S, ref=np.max)

    representation_data = resize(spec, (IMG_HEIGHT,IMG_WIDTH))
    representation_data = normalize_to_range(representation_data)
    
    return representation_data

def normalize_to_range(matrix, new_min=0, new_max=1):
    """
    Normalize the input matrix to a specified range [new_min, new_max].

    Parameters:
    - matrix: Input data.
    - new_min, new_max: The target range for normalization.

    Returns:
    - Normalized data scaled to the range [new_min, new_max].
    """
    original_max = matrix.max()
    original_min = matrix.min()
    # Scale the matrix to [0, 1]
    normalized = (matrix - original_min) / (original_max - original_min)
    # Scale and shift to [new_min, new_max]
    scaled = normalized * (new_max - new_min) + new_min
    
    return scaled
