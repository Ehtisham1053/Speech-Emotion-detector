import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import logging
import json
import seaborn as sns
import time
from tqdm import tqdm

from feature_extractor import process_dataset, analyze_dataset_structure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define emotion classes - modify this based on your dataset
EMOTIONS = {
    'angry': 0,
    'disgust': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

def prepare_data(dataset_path):
    """
    Prepare data for model training
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        X_train, X_test, y_train, y_test: Train and test splits
    """
    logger.info("Extracting features from dataset...")
    
    # First, analyze the dataset structure to help with debugging
    structure = analyze_dataset_structure(dataset_path)
    logger.info(f"Dataset contains {structure['total_files']} files")
    logger.info(f"Found folders: {', '.join(structure['folders'][:10])}")
    
    # Process the dataset
    X, y, file_paths = process_dataset(dataset_path, EMOTIONS)
    
    if len(X) == 0:
        logger.error("No features were extracted. Check your dataset structure and emotion mapping.")
        return None, None, None, None
    
    logger.info(f"Extracted features from {len(X)} files")
    logger.info(f"Feature vector shape: {X.shape}")
    
    # Count samples per emotion
    emotion_counts = {}
    for emotion_idx in np.unique(y):
        emotion_name = list(EMOTIONS.keys())[list(EMOTIONS.values()).index(emotion_idx)]
        count = np.sum(y == emotion_idx)
        emotion_counts[emotion_name] = int(count)
    
    logger.info(f"Samples per emotion: {json.dumps(emotion_counts, indent=2)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train, X_test, y_train, y_test

def train_random_forest_model(X_train, X_test, y_train, y_test):
    """
    Train Random Forest model for speech emotion recognition with hyperparameter tuning
    
    Args:
        X_train, X_test, y_train, y_test: Train and test data
        
    Returns:
        Dictionary with trained model and performance metrics
    """
    logger.info("Training Random Forest model with hyperparameter tuning...")
    
    # First, train a basic model to get a baseline
    logger.info("Training baseline Random Forest model...")
    baseline_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    baseline_model.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
    logger.info(f"Baseline Random Forest accuracy: {baseline_accuracy:.4f}")
    
    # Define hyperparameter grid for tuning
    logger.info("Starting hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    logger.info(f"Best parameters: {best_params}")
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Best Random Forest accuracy: {accuracy:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    class_names = list(EMOTIONS.keys())
    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")
    
    # Save classification report
    with open('models/rf_classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save the best model
    joblib.dump(best_model, 'models/random_forest_model.pkl')
    
    # Save feature importances
    feature_importances = best_model.feature_importances_
    
    return {
        'model': best_model,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'best_params': best_params,
        'feature_importances': feature_importances
    }

def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Random Forest Model')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('models/rf_confusion_matrix.png')
    plt.close()

def plot_feature_importances(feature_importances):
    """
    Plot feature importances
    
    Args:
        feature_importances: Feature importances from the Random Forest model
    """
    # Get indices of top 30 features
    indices = np.argsort(feature_importances)[-30:]
    
    plt.figure(figsize=(12, 8))
    plt.title('Top 30 Feature Importances')
    plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('models/rf_feature_importances.png')
    plt.close()

def main(dataset_path):
    """
    Main function to train the Random Forest model
    
    Args:
        dataset_path: Path to the dataset directory
    """
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(dataset_path)
    
    if X_train is None:
        logger.error("Data preparation failed. Exiting.")
        return None
    
    # Train Random Forest model with hyperparameter tuning
    rf_model = train_random_forest_model(X_train, X_test, y_train, y_test)
    
    # Plot confusion matrix
    class_names = list(EMOTIONS.keys())
    plot_confusion_matrix(rf_model['confusion_matrix'], class_names)
    
    # Plot feature importances
    plot_feature_importances(rf_model['feature_importances'])
    
    # Save emotion mapping
    joblib.dump(EMOTIONS, 'models/emotions.pkl')
    
    logger.info("Random Forest model trained and saved successfully!")
    logger.info(f"Model accuracy: {rf_model['accuracy']:.4f}")
    logger.info(f"Best parameters: {rf_model['best_params']}")
    
    return rf_model['accuracy']

if __name__ == "__main__":
    # Example usage
    print("Speech Emotion Recognition - Random Forest Model Training")
    print("This script trains a Random Forest model on a speech emotion dataset.")
    
    # If a command-line argument is provided, use it as the dataset path
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"Using dataset at: {dataset_path}")
        main(dataset_path)
    else:
        print("Please provide the path to your dataset:")
        print("Example: python train_model.py /path/to/dataset")
        
        # Provide instructions for common datasets
        print("\nDataset download instructions:")
        print("1. RAVDESS: https://zenodo.org/record/1188976")
        print("2. TESS: https://tspace.library.utoronto.ca/handle/1807/24487")
        print("3. CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D")