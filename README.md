# Speech Emotion Recognition (SER) System

This project implements a complete Speech Emotion Recognition system that analyzes speech patterns to identify human emotions such as happy, sad, angry, fearful, neutral, etc.

## Features

- **Audio Feature Extraction**: Extract MFCC, Chroma, and Spectral Contrast features from audio files
- **Multiple Models**: Includes both traditional ML (Random Forest, SVM) and deep learning (LSTM, CNN) models
- **Interactive Web Interface**: Streamlit-based UI for easy interaction
- **Real-time Recording**: Record audio directly from the microphone
- **Visualizations**: Display waveforms, spectrograms, and emotion probabilities

## Project Structure

\`\`\`
speech-emotion-recognition/
├── app.py                  # Streamlit web interface
├── feature_extractor.py    # Audio feature extraction
├── train_model.py          # Model training
├── prediction.py           # Prediction pipeline
├── utils.py                # Utility functions
├── models/                 # Trained models
├── dataset/                # Dataset directory
└── audio_samples/          # Sample audio files
\`\`\`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition