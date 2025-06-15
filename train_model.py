import os
import numpy as np
from cleanliness_model import CleanlinessModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def load_and_prepare_data(train_dir):
    """Load images and prepare features for training."""
    model = CleanlinessModel()
    features = []
    labels = []
    
    # Process clean images
    clean_dir = os.path.join(train_dir, 'clean')
    for img_file in os.listdir(clean_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(clean_dir, img_file)
            try:
                feature_vector = model.extract_features(img_path)
                features.append(feature_vector)
                labels.append(1)  # 1 for clean
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    # Process messy images
    messy_dir = os.path.join(train_dir, 'messy')
    for img_file in os.listdir(messy_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(messy_dir, img_file)
            try:
                feature_vector = model.extract_features(img_path)
                features.append(feature_vector)
                labels.append(0)  # 0 for messy
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    return np.array(features), np.array(labels)

def train_model():
    """Train the cleanliness classification model."""
    print("Loading and preparing training data...")
    train_dir = os.path.join('images', 'train')
    X, y = load_and_prepare_data(train_dir)
    
    if len(X) == 0:
        raise ValueError("No valid images found for training!")
    
    print(f"Training data shape: {X.shape}")
    print(f"Number of clean images: {np.sum(y == 1)}")
    print(f"Number of messy images: {np.sum(y == 0)}")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest classifier
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_score = clf.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.3f}")
    
    # Save the trained model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'cleanliness_classifier.joblib'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model() 