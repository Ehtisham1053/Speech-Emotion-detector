import os
import numpy as np
import joblib
import librosa
import sounddevice as sd
import soundfile as sf
import logging

from feature_extractor import extract_features

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(models_dir='models'):
    """
    Load trained Random Forest model and required files
    
    Args:
        models_dir: Directory containing the models
        
    Returns:
        Dictionary with loaded model and preprocessing objects
    """
    loaded = {}
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        logger.error(f"Models directory '{models_dir}' not found. Please train the model first.")
        return None
    
    # Load emotion mapping
    try:
        loaded['emotions'] = joblib.load(os.path.join(models_dir, 'emotions.pkl'))
        logger.info(f"Loaded emotions mapping: {loaded['emotions']}")
    except Exception as e:
        logger.error(f"Error loading emotions mapping: {str(e)}")
        return None
    
    # Load scaler
    try:
        loaded['scaler'] = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        logger.info("Feature scaler loaded successfully")
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        return None
    
    # Load Random Forest model
    try:
        loaded['model'] = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
        logger.info("Random Forest model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Random Forest model: {str(e)}")
        return None
    
    return loaded

def predict_emotion(audio_path, model_dict=None):
    """
    Predict emotion from audio file
    
    Args:
        audio_path: Path to audio file
        model_dict: Dictionary with loaded model and preprocessing objects
        
    Returns:
        Predicted emotion and confidence score
    """
    # Check if audio file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None
    
    # Load model if not provided
    if model_dict is None:
        model_dict = load_model()
        
    if model_dict is None:
        logger.error("Failed to load model")
        return None
    
    # Extract features
    features_dict = extract_features(audio_path)
    if features_dict is None:
        logger.error(f"Failed to extract features from {audio_path}")
        return None
    
    features = features_dict['features']
    
    # Scale features
    scaled_features = model_dict['scaler'].transform(features.reshape(1, -1))
    
    # Get emotion labels
    emotions = model_dict['emotions']
    emotion_labels = {v: k for k, v in emotions.items()}
    
    # Make prediction
    prediction = model_dict['model'].predict(scaled_features)[0]
    
    # Get prediction probabilities
    probabilities = model_dict['model'].predict_proba(scaled_features)[0]
    confidence = float(probabilities[prediction])
    
    # Get emotion label
    emotion = emotion_labels[prediction]
    
    return {
        'emotion': emotion,
        'confidence': confidence,
        'probabilities': {emotion_labels[i]: float(p) for i, p in enumerate(probabilities)}
    }

def record_audio(duration=5, sample_rate=16000, file_path='audio_samples/recorded_audio.wav'):
    """
    Record audio from microphone
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate
        file_path: Path to save the recorded audio
        
    Returns:
        Path to the saved audio file
    """
    logger.info(f"Recording audio for {duration} seconds...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Save audio
        sf.write(file_path, recording, sample_rate)
        logger.info(f"Audio saved to {file_path}")
        
        return file_path
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    print("Speech Emotion Recognition - Prediction Module")
    print("1. Record audio")
    print("2. Predict from file")
    
    choice = input("Enter your choice (1/2): ")
    
    # Load model
    model_dict = load_model()
    
    if model_dict is None:
        print("Failed to load model. Please train the model first.")
        exit(1)
    
    if choice == '1':
        # Record audio
        audio_file = record_audio()
        if audio_file is None:
            print("Failed to record audio. Please check your microphone.")
            exit(1)
        
    elif choice == '2':
        # Use existing file
        audio_file = input("Enter path to audio file: ")
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            exit(1)
        
    else:
        print("Invalid choice")
        exit(1)
    
    # Make prediction
    result = predict_emotion(audio_file, model_dict=model_dict)
    
    if result:
        # Print result
        print(f"\nPredicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Print all probabilities
        print("\nProbabilities for all emotions:")
        for emotion, prob in result['probabilities'].items():
            print(f"{emotion}: {prob:.4f}")
    else:
        print("Prediction failed.")