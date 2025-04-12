import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_audio(file_path, target_sr=16000, mono=True, normalize=True):
    """
    Preprocess audio file: convert to WAV, resample, convert to mono, normalize
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize audio if True
        
    Returns:
        Path to preprocessed audio file
    """
    # Create output directory
    output_dir = os.path.join(os.path.dirname(file_path), 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"proc_{filename}")
    
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=mono)
        
        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Normalize if needed
        if normalize:
            y = librosa.util.normalize(y)
        
        # Save as WAV
        sf.write(output_path, y, target_sr)
        
        logger.info(f"Preprocessed {file_path} -> {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error preprocessing {file_path}: {str(e)}")
        return None

def batch_preprocess(input_dir, target_sr=16000):
    """
    Batch preprocess all audio files in a directory
    
    Args:
        input_dir: Input directory
        target_sr: Target sample rate
        
    Returns:
        List of preprocessed file paths
    """
    processed_files = []
    skipped_files = 0
    
    logger.info(f"Batch preprocessing files in {input_dir}")
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(root, file)
                processed_path = preprocess_audio(file_path, target_sr=target_sr)
                
                if processed_path:
                    processed_files.append(processed_path)
                else:
                    skipped_files += 1
    
    logger.info(f"Preprocessing complete: {len(processed_files)} files processed, {skipped_files} files skipped")
    return processed_files

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.tight_layout()
    
    return plt.gcf()

def create_dataset_summary(dataset_path):
    """
    Create a summary of the dataset
    
    Args:
        dataset_path: Path to dataset
        
    Returns:
        Dictionary with dataset summary
    """
    summary = {
        'total_files': 0,
        'emotions': {},
        'duration': 0,
        'sample_rates': set(),
        'channels': set()
    }
    
    logger.info(f"Creating summary for dataset at {dataset_path}")
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # Count files
                summary['total_files'] += 1
                
                # Extract emotion from filename or directory
                emotion = None
                
                # Try to extract from parent folder
                parent_folder = os.path.basename(root)
                
                # Example for nested directory structure
                if parent_folder.lower() in ['angry', 'happy', 'sad', 'fearful', 'neutral', 'surprised', 'disgust']:
                    emotion = parent_folder.lower()
                
                # Example for RAVDESS dataset
                if emotion is None and 'RAVDESS' in dataset_path:
                    try:
                        emotion_code = file.split('-')[2]
                        emotion_map = {
                            '01': 'neutral',
                            '02': 'calm',
                            '03': 'happy',
                            '04': 'sad',
                            '05': 'angry',
                            '06': 'fearful',
                            '07': 'disgust',
                            '08': 'surprised'
                        }
                        if emotion_code in emotion_map:
                            emotion = emotion_map[emotion_code]
                    except:
                        pass
                
                # Example for TESS dataset
                if emotion is None and 'TESS' in dataset_path:
                    if '_angry' in file:
                        emotion = 'angry'
                    elif '_happy' in file:
                        emotion = 'happy'
                    elif '_sad' in file:
                        emotion = 'sad'
                    elif '_fear' in file:
                        emotion = 'fearful'
                    elif '_disgust' in file:
                        emotion = 'disgust'
                    elif '_neutral' in file:
                        emotion = 'neutral'
                    elif '_ps' in file:
                        emotion = 'surprised'
                
                # Count emotions
                if emotion:
                    if emotion in summary['emotions']:
                        summary['emotions'][emotion] += 1
                    else:
                        summary['emotions'][emotion] = 1
                
                # Get audio info
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    summary['duration'] += duration
                    summary['sample_rates'].add(sr)
                    
                    # Check channels
                    if len(y.shape) > 1:
                        summary['channels'].add(y.shape[1])
                    else:
                        summary['channels'].add(1)
                except Exception as e:
                    logger.warning(f"Error analyzing {file_path}: {str(e)}")
    
    # Convert sets to lists for JSON serialization
    summary['sample_rates'] = list(summary['sample_rates'])
    summary['channels'] = list(summary['channels'])
    
    logger.info(f"Dataset summary created: {summary['total_files']} files, {len(summary['emotions'])} emotions")
    return summary

if __name__ == "__main__":
    # Example usage
    print("Utility functions for Speech Emotion Recognition")
    
    # Example: Preprocess a single audio file
    if os.path.exists('sample.wav'):
        processed_file = preprocess_audio('sample.wav')
        print(f"Preprocessed file: {processed_file}")
    
    # If a dataset path is provided, create a summary
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"Creating summary for dataset at: {dataset_path}")
        summary = create_dataset_summary(dataset_path)
        
        print(f"Total files: {summary['total_files']}")
        print(f"Total duration: {summary['duration'] / 3600:.2f} hours")
        print(f"Emotions found: {summary['emotions']}")
        print(f"Sample rates: {summary['sample_rates']}")
        print(f"Channels: {summary['channels']}")