import numpy as np
from PIL import Image
import io
from skimage.feature import hog
from skimage import color, transform
import os
import json
from sklearn.ensemble import RandomForestClassifier
import joblib

class CleanlinessModel:
    def __init__(self):
        self.model_path = os.path.join('models', 'cleanliness_classifier.joblib')
        self.classifier = self._load_model()
        if self.classifier is None:
            self._initialize_default_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return None
        return None
    
    def _initialize_default_model(self):
        """Initialize a default RandomForestClassifier with reasonable parameters"""
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        # Save the default model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.classifier, self.model_path)
    
    def _preprocess_image(self, image_path):
        try:
            # Load and convert to grayscale
            image = Image.open(image_path).convert('L')
            
            # Resize to standard size
            image = image.resize((128, 128))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def extract_features(self, image_path):
        """Extract features from an image for training or prediction."""
        # Preprocess image
        image_array = self._preprocess_image(image_path)
        
        # Calculate basic image metrics
        brightness = np.mean(image_array) / 255.0
        contrast = np.std(image_array) / 255.0
        
        # Calculate edge intensity
        gradient_x = np.gradient(image_array, axis=0)
        gradient_y = np.gradient(image_array, axis=1)
        edge_intensity = np.mean(np.sqrt(gradient_x**2 + gradient_y**2)) / 255.0
        
        # Calculate HOG features
        hog_features = hog(
            image_array,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=False
        )
        
        # Combine all features
        features = np.concatenate([
            [brightness, contrast, edge_intensity],
            hog_features
        ])
        
        return features
    
    def predict(self, image_path):
        """Predict cleanliness of an image."""
        if self.classifier is None:
            raise Exception("Model not trained yet!")
        
        features = self.extract_features(image_path)
        prediction = self.classifier.predict([features])[0]
        probability = self.classifier.predict_proba([features])[0]
        
        return {
            'prediction': 'Clean' if prediction == 1 else 'Messy',
            'confidence': float(max(probability)),
            'features': features.tolist()
        }

class CleanlinessPredictor:
    def __init__(self):
        self.model_path = 'cleanliness_model.json'
        self.thresholds = self._load_or_create_thresholds()
        
    def _load_or_create_thresholds(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default thresholds if no model exists
        return {
            'brightness_threshold': 0.6,
            'contrast_threshold': 0.4,
            'edge_threshold': 0.3
        }
    
    def _preprocess_image(self, image_bytes):
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale and resize
            image = image.convert('L')
            image = image.resize((128, 128))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def _calculate_metrics(self, image_array):
        # Calculate brightness (mean pixel value)
        brightness = np.mean(image_array) / 255.0
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(image_array) / 255.0
        
        # Calculate edge intensity using simple gradient
        gradient_x = np.gradient(image_array, axis=0)
        gradient_y = np.gradient(image_array, axis=1)
        edge_intensity = np.mean(np.sqrt(gradient_x**2 + gradient_y**2)) / 255.0
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'edge_intensity': edge_intensity
        }
    
    def predict(self, image_bytes):
        try:
            # Preprocess image
            image_array = self._preprocess_image(image_bytes)
            
            # Calculate metrics
            metrics = self._calculate_metrics(image_array)
            
            # Calculate cleanliness score based on metrics
            brightness_score = metrics['brightness'] / self.thresholds['brightness_threshold']
            contrast_score = metrics['contrast'] / self.thresholds['contrast_threshold']
            edge_score = 1 - (metrics['edge_intensity'] / self.thresholds['edge_threshold'])
            
            # Combine scores with weights
            score = (0.4 * brightness_score + 0.3 * contrast_score + 0.3 * edge_score)
            score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            
            # Determine classification
            classification = "Clean" if score > 0.7 else "Needs Attention" if score > 0.3 else "Dirty"
            
            return {
                'score': score,
                'classification': classification,
                'metrics': metrics
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def update_thresholds(self, clean_images, dirty_images):
        """
        Update thresholds based on labeled images
        clean_images: List of image bytes for clean rooms
        dirty_images: List of image bytes for dirty rooms
        """
        try:
            clean_metrics = []
            dirty_metrics = []
            
            # Process clean images
            for img in clean_images:
                image_array = self._preprocess_image(img)
                metrics = self._calculate_metrics(image_array)
                clean_metrics.append(metrics)
            
            # Process dirty images
            for img in dirty_images:
                image_array = self._preprocess_image(img)
                metrics = self._calculate_metrics(image_array)
                dirty_metrics.append(metrics)
            
            # Calculate new thresholds
            if clean_metrics and dirty_metrics:
                clean_brightness = np.mean([m['brightness'] for m in clean_metrics])
                dirty_brightness = np.mean([m['brightness'] for m in dirty_metrics])
                self.thresholds['brightness_threshold'] = (clean_brightness + dirty_brightness) / 2
                
                clean_contrast = np.mean([m['contrast'] for m in clean_metrics])
                dirty_contrast = np.mean([m['contrast'] for m in dirty_metrics])
                self.thresholds['contrast_threshold'] = (clean_contrast + dirty_contrast) / 2
                
                clean_edge = np.mean([m['edge_intensity'] for m in clean_metrics])
                dirty_edge = np.mean([m['edge_intensity'] for m in dirty_metrics])
                self.thresholds['edge_threshold'] = (clean_edge + dirty_edge) / 2
                
                # Save updated thresholds
                with open(self.model_path, 'w') as f:
                    json.dump(self.thresholds, f)
                
                return True
            
            return False
            
        except Exception as e:
            return {'error': str(e)}

def train_model(train_data_path, model_save_path, num_epochs=10, batch_size=32):
    """Train the cleanliness classifier
    
    Args:
        train_data_path (str): Path to directory containing training data
            Expected structure:
            train_data_path/
                clean/
                    image1.jpg
                    image2.jpg
                    ...
                dirty/
                    image1.jpg
                    image2.jpg
                    ...
        model_save_path (str): Path to save the trained model
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    try:
        # Initialize model
        model = CleanlinessModel()
        
        # Load and preprocess training data
        X = []  # Features
        y = []  # Labels
        
        # Process clean images
        clean_dir = os.path.join(train_data_path, 'clean')
        if os.path.exists(clean_dir):
            for img_file in os.listdir(clean_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(clean_dir, img_file)
                    try:
                        features = model.extract_features(img_path)
                        X.append(features)
                        y.append(1)  # 1 for clean
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
        
        # Process dirty images
        dirty_dir = os.path.join(train_data_path, 'dirty')
        if os.path.exists(dirty_dir):
            for img_file in os.listdir(dirty_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dirty_dir, img_file)
                    try:
                        features = model.extract_features(img_path)
                        X.append(features)
                        y.append(0)  # 0 for dirty
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
        
        if not X or not y:
            raise Exception("No valid training data found")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Train the model
        model.classifier.fit(X, y)
        
        # Save the trained model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model.classifier, model_save_path)
        
        return True
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

# Example usage:
if __name__ == "__main__":
    # Initialize predictor
    predictor = CleanlinessPredictor()
    
    # Example prediction
    with open('example_image.jpg', 'rb') as f:
        image_bytes = f.read()
        result = predictor.predict(image_bytes)
        print(result) 