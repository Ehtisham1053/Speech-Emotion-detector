import os
import sys
import logging
from feature_extractor import analyze_dataset_structure

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(dataset_path):
    """
    Analyze the dataset structure and print detailed information
    
    Args:
        dataset_path: Path to the dataset directory
    """
    logger.info(f"Analyzing dataset structure at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return
    
    # Analyze dataset structure
    structure = analyze_dataset_structure(dataset_path)
    
    # Print basic information
    logger.info(f"Total files found: {structure['total_files']}")
    logger.info(f"File extensions: {', '.join(structure['file_extensions'])}")
    
    # Print folder structure
    logger.info("Folder structure:")
    for folder in sorted(structure['folders'])[:20]:  # Limit to first 20 folders
        logger.info(f"  {folder}")
    
    if len(structure['folders']) > 20:
        logger.info(f"  ... and {len(structure['folders']) - 20} more folders")
    
    # Print sample folder names
    logger.info("Sample folder names:")
    for folder in structure['sample_folders']:
        logger.info(f"  {folder}")
    
    # Print sample filenames
    logger.info("Sample filenames:")
    for filename in structure['sample_filenames'][:10]:
        logger.info(f"  {filename}")
    
    # Check for emotion keywords in folder names
    emotions = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    logger.info("Checking for emotion keywords in folder names:")
    
    emotion_found = False
    for folder in structure['sample_folders']:
        folder_lower = folder.lower()
        found_emotions = [emotion for emotion in emotions if emotion in folder_lower]
        if found_emotions:
            logger.info(f"  Folder '{folder}' contains emotions: {', '.join(found_emotions)}")
            emotion_found = True
    
    if not emotion_found:
        logger.warning("No emotion keywords found in folder names. Check if emotions are encoded differently.")
    
    # Provide guidance
    logger.info("\nGuidance for your dataset:")
    if emotion_found:
        logger.info("Your dataset appears to have emotions encoded in folder names.")
        logger.info("The updated feature_extractor.py should be able to process this structure.")
        logger.info("Try running: python train_model.py " + dataset_path)
    else:
        logger.info("Could not automatically detect how emotions are encoded in your dataset.")
        logger.info("You may need to manually inspect the files and folders to determine the pattern.")
        logger.info("Then modify the process_dataset() function in feature_extractor.py accordingly.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        main(dataset_path)
    else:
        print("Please provide the path to your dataset:")
        print("Example: python analyze_dataset.py /path/to/dataset")