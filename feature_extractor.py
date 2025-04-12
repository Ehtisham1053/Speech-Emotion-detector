import librosa
import numpy as np
import os
from tqdm import tqdm
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features(file_path, mfcc_len=40):
    """
    Extract audio features from a file:
    - MFCC (Mel Frequency Cepstral Coefficients)
    - Chroma
    - Spectral Contrast
    
    Args:
        file_path: Path to audio file
        mfcc_len: Number of MFCC features to extract
        
    Returns:
        Dictionary containing the extracted features
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=16000)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Extract features
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_len)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        
        # 3. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        spectral_contrast_std = np.std(spectral_contrast.T, axis=0)
        
        # 4. Additional features for better performance
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Root Mean Square Energy
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)
        
        # Combine all features
        features = np.concatenate((
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            spectral_contrast_mean, spectral_contrast_std,
            [zcr_mean, zcr_std, rmse_mean, rmse_std, spectral_centroid_mean, spectral_centroid_std]
        ))
        
        return {
            'features': features,
            'mfcc': mfcc,
            'chroma': chroma,
            'spectral_contrast': spectral_contrast,
            'zcr': zcr,
            'rmse': rmse,
            'spectral_centroid': spectral_centroid
        }
    
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_dataset(dataset_path, emotions_map):
    """
    Process all audio files in the dataset and extract features
    
    Args:
        dataset_path: Path to the dataset directory
        emotions_map: Dictionary mapping emotion labels to numerical values
        
    Returns:
        X: Features array
        y: Labels array
        file_paths: List of processed file paths
    """
    X = []
    y = []
    file_paths = []
    
    # Count files for statistics
    total_files = 0
    processed_files = 0
    skipped_files = 0
    emotion_counts = {emotion: 0 for emotion in set(emotions_map.values())}
    
    logger.info(f"Processing dataset at: {dataset_path}")
    
    # Special case mappings for TESS dataset
    folder_mappings = {
        'oaf_fear': 'fearful',
        'yaf_fear': 'fearful',
        'oaf_pleasant_surprise': 'surprised',
        'yaf_pleasant_surprise': 'surprised'
    }
    
    file_suffix_mappings = {
        '_fear': 'fearful',
        '_ps': 'surprised'
    }
    
    # Walk through the dataset directory
    for root, dirs, files in tqdm(os.walk(dataset_path), desc="Processing dataset"):
        for file in files:
            if file.endswith('.wav'):
                total_files += 1
                file_path = os.path.join(root, file)
                
                # Extract emotion from directory structure or filename
                emotion = None
                
                # APPROACH 1: Check for exact folder name matches with special cases
                folder_name = os.path.basename(root).lower()
                
                # Direct mapping for special folder names
                if folder_name in folder_mappings:
                    emotion = folder_mappings[folder_name]
                    logger.debug(f"Mapped folder '{folder_name}' to emotion '{emotion}'")
                
                # APPROACH 2: Check for standard emotion names in folder
                if emotion is None:
                    for emotion_name in emotions_map.keys():
                        if emotion_name.lower() == folder_name.lower():
                            emotion = emotion_name
                            break
                
                # APPROACH 3: Check for emotion names contained within folder name
                if emotion is None:
                    for emotion_name in emotions_map.keys():
                        if emotion_name.lower() in folder_name.lower():
                            emotion = emotion_name
                            break
                
                # APPROACH 4: Check file suffix mappings
                if emotion is None:
                    file_lower = file.lower()
                    for suffix, mapped_emotion in file_suffix_mappings.items():
                        if suffix in file_lower:
                            emotion = mapped_emotion
                            break
                
                # APPROACH 5: Check for standard emotion names in filename
                if emotion is None:
                    file_lower = file.lower()
                    for emotion_name in emotions_map.keys():
                        if emotion_name.lower() in file_lower:
                            emotion = emotion_name
                            break
                
                # Special case for files with "fear" in the name
                if emotion is None and "_fear" in file.lower():
                    emotion = "fearful"
                
                # Skip if emotion couldn't be determined
                if emotion is None or emotion not in emotions_map:
                    logger.warning(f"Skipping {file_path}: Could not determine emotion")
                    skipped_files += 1
                    continue
                
                # Extract features
                features_dict = extract_features(file_path)
                if features_dict is not None:
                    X.append(features_dict['features'])
                    y.append(emotions_map[emotion])
                    file_paths.append(file_path)
                    processed_files += 1
                    emotion_counts[emotions_map[emotion]] += 1
                else:
                    skipped_files += 1
    
    # Log statistics
    logger.info(f"Dataset processing complete:")
    logger.info(f"  Total files found: {total_files}")
    logger.info(f"  Files processed successfully: {processed_files}")
    logger.info(f"  Files skipped: {skipped_files}")
    
    # Convert emotion counts to emotion names
    emotion_name_counts = {}
    for emotion_idx, count in emotion_counts.items():
        # Find all emotions that map to this index
        emotion_names = [k for k, v in emotions_map.items() if v == emotion_idx]
        if emotion_names:
            emotion_name = '/'.join(set(emotion_names))  # Join multiple names with '/'
            emotion_name_counts[emotion_name] = count
    
    logger.info(f"  Emotion distribution: {json.dumps(emotion_name_counts, indent=2)}")
    
    if len(X) == 0:
        logger.error("No files were processed successfully. Check your dataset structure and emotion mapping.")
        return np.array([]), np.array([]), []
    
    return np.array(X), np.array(y), file_paths

def visualize_audio(file_path):
    """
    Generate visualizations for an audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio data and features for visualization
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=16000)
    
    # Extract features for visualization
    features_dict = extract_features(file_path)
    
    # Create visualization data
    duration = librosa.get_duration(y=y, sr=sr)
    time_axis = np.linspace(0, duration, len(y))
    
    # Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Chromagram
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    return {
        'waveform': (time_axis, y),
        'mel_spectrogram': mel_spectrogram_db,
        'chromagram': chromagram,
        'mfcc': mfcc,
        'features': features_dict
    }

def analyze_dataset_structure(dataset_path):
    """
    Analyze the dataset structure to help identify how emotions are encoded
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary with dataset structure information
    """
    structure = {
        'total_files': 0,
        'folders': set(),
        'subfolders': {},
        'file_extensions': set(),
        'sample_filenames': [],
        'sample_folders': []
    }
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        # Get relative path from dataset_path
        rel_path = os.path.relpath(root, dataset_path)
        if rel_path != '.':
            # Add to folders list
            structure['folders'].add(rel_path)
            
            # Track subfolders
            parent = os.path.dirname(rel_path)
            if parent not in structure['subfolders']:
                structure['subfolders'][parent] = []
            if os.path.basename(rel_path) not in structure['subfolders'][parent]:
                structure['subfolders'][parent].append(os.path.basename(rel_path))
            
            # Sample some folder names
            if len(structure['sample_folders']) < 20:
                structure['sample_folders'].append(os.path.basename(root))
        
        # Process files
        for file in files:
            structure['total_files'] += 1
            
            # Track file extensions
            ext = os.path.splitext(file)[1]
            structure['file_extensions'].add(ext)
            
            # Sample some filenames
            if len(structure['sample_filenames']) < 20:
                structure['sample_filenames'].append(file)
    
    # Convert sets to lists for easier reading
    structure['folders'] = sorted(list(structure['folders']))
    structure['file_extensions'] = sorted(list(structure['file_extensions']))
    
    return structure

# Example usage
if __name__ == "__main__":
    # Test feature extraction on a sample file
    sample_file = "sample.wav"
    
    # Create a sample audio file if it doesn't exist
    if not os.path.exists(sample_file):
        import scipy.io.wavfile as wav
        
        # Generate a simple sine wave
        sr = 16000
        t = np.linspace(0, 3, sr * 3)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        wav.write(sample_file, sr, (audio * 32767).astype(np.int16))
        print(f"Created sample audio file: {sample_file}")
    
    # Extract features
    features = extract_features(sample_file)
    print(f"Extracted features shape: {features['features'].shape}")
    
    # If a dataset path is provided, analyze its structure
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"\nAnalyzing dataset structure at: {dataset_path}")
        structure = analyze_dataset_structure(dataset_path)
        
        print(f"Total files: {structure['total_files']}")
        print(f"File extensions: {', '.join(structure['file_extensions'])}")
        print(f"Top-level folders: {', '.join(structure['subfolders'].get('.', []))}")
        print(f"Sample filenames: {', '.join(structure['sample_filenames'][:5])}")
        print(f"Sample folder names: {', '.join(structure['sample_folders'][:10])}")
        
        print("\nTo identify how emotions are encoded in your dataset, examine the folder names and file patterns.")
