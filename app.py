import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import time
import matplotlib.pyplot as plt
import joblib
from feature_extractor import extract_features, visualize_audio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define emotion mapping (ensure this matches what was used in training)
EMOTIONS = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised"
}

# Function to load the trained model
@st.cache_resource
def load_model():
    try:
        # Load Random Forest model
        model_path = "models/random_forest_model.pkl"
        if not os.path.exists(model_path):
            return None
        
        model = joblib.load(model_path)
        logger.info("Random Forest model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Function to make predictions
def predict_emotion(audio_path, model):
    try:
        # Extract features
        features_dict = extract_features(audio_path)
        if features_dict is None:
            return None, None
        
        # Get features array for prediction
        features = features_dict['features'].reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return prediction, probabilities
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
        return None, None

# Function to record audio
def record_audio(duration=5, sample_rate=16000):
    try:
        st.write("Recording...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(duration/100)
            progress_bar.progress(i + 1)
        
        sd.wait()
        st.write("Recording complete!")
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio_data, sample_rate)
        
        return temp_file.name
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        st.error(f"Error recording audio: {str(e)}")
        return None

# Function to display audio visualization
def display_audio_visualization(audio_path):
    try:
        # Get visualization data
        viz_data = visualize_audio(audio_path)
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot waveform
        time_axis, waveform = viz_data['waveform']
        axs[0].plot(time_axis, waveform, color='blue')
        axs[0].set_title('Waveform')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        
        # Plot mel spectrogram
        mel_spec = viz_data['mel_spectrogram']
        img = axs[1].imshow(mel_spec, aspect='auto', origin='lower', interpolation='none')
        axs[1].set_title('Mel Spectrogram')
        axs[1].set_ylabel('Mel Bands')
        axs[1].set_xlabel('Time')
        fig.colorbar(img, ax=axs[1], format='%+2.0f dB')
        
        # Plot MFCC
        mfcc = viz_data['mfcc']
        img = axs[2].imshow(mfcc, aspect='auto', origin='lower', interpolation='none')
        axs[2].set_title('MFCC')
        axs[2].set_ylabel('MFCC Coefficients')
        axs[2].set_xlabel('Time')
        fig.colorbar(img, ax=axs[2])
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        logger.error(f"Error displaying audio visualization: {str(e)}")
        st.error(f"Error displaying audio visualization: {str(e)}")

# Function to display prediction results
def display_prediction_results(prediction, probabilities):
    if prediction is None:
        st.error("Could not make a prediction. Please try again.")
        return
    
    # Display the predicted emotion
    emotion = EMOTIONS[prediction]
    st.success(f"Predicted Emotion: {emotion.upper()}")
    
    # Display probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    emotions = [EMOTIONS[i] for i in range(len(EMOTIONS))]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#CCCCCC', '#C2C2F0', '#FFD700']
    
    # Create horizontal bar chart
    y_pos = np.arange(len(emotions))
    ax.barh(y_pos, probabilities, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e.capitalize() for e in emotions])
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Probability Distribution')
    
    # Add percentage labels to the bars
    for i, v in enumerate(probabilities):
        ax.text(v + 0.01, i, f"{v:.2%}", va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

# Main app
def main():
    st.set_page_config(
        page_title="Speech Emotion Recognition",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Speech Emotion Recognition")
    st.write("""
    This application analyzes speech audio to detect emotions using a Random Forest model.
    Upload an audio file or record your voice to see the emotion prediction in real-time.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("""
        Random Forest model not found. Please train the model first using the train_model.py script.
        
        Run: `python train_model.py /path/to/dataset`
        """)
        
        # Dataset structure analyzer
        st.header("Dataset Structure Analyzer")
        dataset_path = st.text_input("Enter path to your dataset:")
        if dataset_path and os.path.exists(dataset_path):
            from feature_extractor import analyze_dataset_structure
            structure = analyze_dataset_structure(dataset_path)
            
            st.write(f"Total files: {structure['total_files']}")
            st.write(f"File extensions: {', '.join(structure['file_extensions'])}")
            
            st.subheader("Sample folder names:")
            for folder in structure['sample_folders'][:10]:
                st.write(f"- {folder}")
                
            st.subheader("Sample filenames:")
            for file in structure['sample_filenames'][:10]:
                st.write(f"- {file}")
        return
    
    # Create two columns for the input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Option 1: Upload Audio")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
        
        if uploaded_file is not None:
            # Save uploaded file to temp location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(uploaded_file.read())
            audio_path = temp_file.name
            
            st.audio(uploaded_file, format='audio/wav')
            
            # Make prediction
            with st.spinner('Analyzing audio...'):
                prediction, probabilities = predict_emotion(audio_path, model)
            
            # Display results
            display_prediction_results(prediction, probabilities)
            
            # Display visualizations
            st.subheader("Audio Visualization")
            display_audio_visualization(audio_path)
            
            # Clean up
            os.unlink(audio_path)
    
    with col2:
        st.header("Option 2: Record Audio")
        duration = st.slider("Recording duration (seconds)", min_value=1, max_value=10, value=5)
        
        if st.button("Record"):
            audio_path = record_audio(duration=duration)
            
            if audio_path:
                st.audio(audio_path, format='audio/wav')
                
                # Make prediction
                with st.spinner('Analyzing audio...'):
                    prediction, probabilities = predict_emotion(audio_path, model)
                
                # Display results
                display_prediction_results(prediction, probabilities)
                
                # Display visualizations
                st.subheader("Audio Visualization")
                display_audio_visualization(audio_path)
                
                # Clean up
                os.unlink(audio_path)
    
    # Add information about the model
    st.sidebar.header("About the Model")
    st.sidebar.write("""
    This application uses a Random Forest classifier to predict emotions from speech.
    
    The model analyzes various audio features including:
    - MFCC (Mel Frequency Cepstral Coefficients)
    - Chroma features
    - Spectral contrast
    - Zero crossing rate
    - Root mean square energy
    - Spectral centroid
    
    The model can detect 7 emotions:
    - Angry
    - Disgusted
    - Fearful
    - Happy
    - Neutral
    - Sad
    - Surprised
    """)
    
    # Add information about the dataset
    st.sidebar.header("Dataset")
    st.sidebar.write("""
    The model was trained on a dataset of audio recordings labeled with emotions.
    
    For best results, use clear audio recordings with minimal background noise.
    """)

if __name__ == "__main__":
    main()
