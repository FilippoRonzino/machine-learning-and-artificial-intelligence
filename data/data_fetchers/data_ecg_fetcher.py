import wfdb
import numpy as np
import pandas as pd

def extract_signals(record_name: str, channel: int) -> np.ndarray:
    """
    Extract ECG signal data from a specific record and channel. It needs in the folder data/ecg_data the ecg data from PhysioNet.

    :param record_name: Name of the ECG record to load.
    :param channel: Index of the channel to extract from the record.
    :return: signal: The extracted signal data as a numpy array.
    """
    try:
        record = wfdb.rdrecord(f"data/ecg_data/{record_name}")
        if channel < 0 or channel >= record.p_signal.shape[1]:
            raise ValueError(f"Channel {channel} does not exist in the record {record_name}.")
        signal = record.p_signal[:, channel]
        return signal
    except FileNotFoundError:
        raise FileNotFoundError(f"Record file {record_name} not found in data/ecg_data.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting signals: {e}")


def downsample_signal(signal: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a signal by selecting every nth sample.

    :param signal: The input signal to downsample.
    :param factor: The downsampling factor (selects every 'factor' sample).
    :return: The downsampled signal as a numpy array.
    """
    if factor <= 0:
        raise ValueError(f"Downsample factor must be a positive integer, got {factor}")
    
    return signal[::factor]


def create_downsampled_signals_array(record_names: list, channel: int, downsample_factor: int) -> np.ndarray:
    """
    Create an array of downsampled signals from multiple ECG records. It needs in the folder data/ecg_data the ecg data from PhysioNet.

    :param record_names: List of ECG record names to process.
    :param channel: Index of the channel to extract from each record.
    :param downsample_factor: Factor by which to downsample each signal.
    :return: downsampled_signals: Array of downsampled signals, truncated to the length of the shortest signal.
    """
    if not record_names:
        raise ValueError("List of record names cannot be empty")
    
    downsampled_signals = []
    min_length = float('inf')  
    
    for record_name in record_names:
        signal = extract_signals(record_name, channel)
        signal_downsampled = downsample_signal(signal, downsample_factor)
        downsampled_signals.append(signal_downsampled)
        min_length = min(min_length, len(signal_downsampled))  
    
    if min_length == 0:
        raise ValueError("One or more signals has zero length after downsampling")
        
    downsampled_signals = [signal[:min_length] for signal in downsampled_signals]
    
    return np.array(downsampled_signals)


def sample_random_segments_across_patients(signals: np.ndarray, n: int, sample_length: int = 80, 
                                           threshold: float = 0.1, max_attempts: int = 10000) -> np.ndarray:
    """
    Sample random segments of a specified length from multiple patient signals.

    :param signals: Array of signals from multiple patients.
    :param n: Number of segments to sample.
    :param sample_length: Length of each segment to sample.
    :param threshold: Minimum sum of absolute values required for a segment to be considered active.
    :param max_attempts: Maximum number of sampling attempts before giving up.
    :return: segments: Array of sampled signal segments.
    """
    if sample_length <= 0:
        raise ValueError(f"Sample length must be a positive integer, got {sample_length}")
    
    if threshold < 0:
        raise ValueError(f"Threshold must be non-negative, got {threshold}")
    
    n_patients, total_length = signals.shape
    
    if total_length < sample_length:
        raise ValueError(f"Signal length ({total_length}) is shorter than the requested sample length ({sample_length})")
    
    segments = []
    attempts = 0
    
    while len(segments) < n and attempts < max_attempts:
        attempts += 1
        
        patient_idx = np.random.randint(0, n_patients)
        start_index = np.random.randint(0, total_length - sample_length + 1)
        segment = signals[patient_idx, start_index:start_index + sample_length]
        
        if np.sum(np.abs(segment)) > threshold:
            segments.append(segment)
    
    if len(segments) < n:
        collected = len(segments)
        raise ValueError(f"Not enough active segments found (only {collected}/{n}). Consider lowering the threshold (current: {threshold}) or increasing max_attempts.")
    
    return np.array(segments)

def create_ecg_datasets(train_patients: list, val_patients: list, test_patients: list, channel: int, 
                        downsample_factor: int, total_segments: int, sample_length: int = 80, 
                        threshold: float = 20, max_attempts: int = 10000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training, validation and testing datasets from ECG signals with exact percentage splits.
    It needs in the folder data/ecg_data the ecg data from PhysioNet.

    :param train_patients: List of patient record names for training data.
    :param val_patients: List of patient record names for validation data.
    :param test_patients: List of patient record names for testing data.
    :param channel: Index of the channel to extract from each record.
    :param downsample_factor: Factor by which to downsample each signal.
    :param total_segments: Total number of segments to create across all datasets.
    :param sample_length: Length of each segment to sample.
    :param threshold: Minimum sum of absolute values required for a segment to be considered active.
    :param max_attempts: Maximum number of sampling attempts before giving up.
    :return: train_segments, val_segments, test_segments: Arrays of sampled signal segments.
    """
    if not train_patients:
        raise ValueError("Train patients list cannot be empty")
    if not val_patients:
        raise ValueError("Validation patients list cannot be empty")
    if not test_patients:
        raise ValueError("Test patients list cannot be empty")
    
    if downsample_factor <= 0:
        raise ValueError(f"Downsample factor must be a positive integer, got {downsample_factor}")
    
    if sample_length <= 0:
        raise ValueError(f"Sample length must be a positive integer, got {sample_length}")
    
    if threshold < 0:
        raise ValueError(f"Threshold must be non-negative, got {threshold}")
    
    # Calculate exact 80-10-10 split
    n_train_segments = int(total_segments * 0.8)
    n_val_segments = int(total_segments * 0.1)
    n_test_segments = total_segments - n_train_segments - n_val_segments  
    
    print(f"Segment split: Train={n_train_segments} ({n_train_segments/total_segments:.1%}), "
          f"Val={n_val_segments} ({n_val_segments/total_segments:.1%}), "
          f"Test={n_test_segments} ({n_test_segments/total_segments:.1%})")
    
    train_signals = create_downsampled_signals_array(train_patients, channel, downsample_factor)
    val_signals = create_downsampled_signals_array(val_patients, channel, downsample_factor)
    test_signals = create_downsampled_signals_array(test_patients, channel, downsample_factor)
    
    _, train_length = train_signals.shape
    _, val_length = val_signals.shape
    _, test_length = test_signals.shape
    
    if train_length < sample_length:
        raise ValueError(f"Training signals length ({train_length}) is shorter than the requested sample length ({sample_length})")
    
    if val_length < sample_length:
        raise ValueError(f"Validation signals length ({val_length}) is shorter than the requested sample length ({sample_length})")
    
    if test_length < sample_length:
        raise ValueError(f"Testing signals length ({test_length}) is shorter than the requested sample length ({sample_length})")
    
    train_segments = sample_random_segments_across_patients(train_signals, n_train_segments, 
                                                            sample_length, threshold, max_attempts)
    
    val_segments = sample_random_segments_across_patients(val_signals, n_val_segments, 
                                                         sample_length, threshold, max_attempts)
    
    test_segments = sample_random_segments_across_patients(test_signals, n_test_segments, 
                                                           sample_length, threshold, max_attempts)
    
    return train_segments, val_segments, test_segments

if __name__ == "__main__":
    train_patients = ['19090', '19088', '18184', '18177', '17453', '17052', '16795', '16786', '16773', '16539', '16483', '16420', '16273', '16272', '16265']
    test_patients = ['19140', '19093']
    val_patients = ['19830']
    channel = 0
    downsample_factor = 8
    sample_length = 80
    threshold = 10
    
    total_segments = 62500  
    
    print("Creating ECG datasets with 80-10-10 train-val-test split...")
    train_segments, val_segments, test_segments = create_ecg_datasets(
        train_patients, val_patients, test_patients, channel, downsample_factor,
        total_segments, sample_length=sample_length, threshold=threshold, max_attempts=100000
    )
    
    print("Train segments shape:", train_segments.shape)
    print("Validation segments shape:", val_segments.shape)
    print("Test segments shape:", test_segments.shape)
    
    print("Saving datasets to parquet files...")
    train_segments_df = pd.DataFrame(train_segments)
    val_segments_df = pd.DataFrame(val_segments)
    test_segments_df = pd.DataFrame(test_segments)
    
    train_segments_df.to_parquet('data/data_storage/ecg_parquets/train_ecg.parquet')
    val_segments_df.to_parquet('data/data_storage/ecg_parquets/val_ecg.parquet')
    test_segments_df.to_parquet('data/data_storage/ecg_parquets/test_ecg.parquet')
    
    print("Datasets saved successfully.")