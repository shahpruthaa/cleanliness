import numpy as np
from PIL import Image
import io
from skimage.feature import hog
from skimage import color, transform
import os
import json

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
    """Train the cleanliness classifier"""
    # This is a placeholder for the training function
    # You'll need to implement data loading and training loop
    # based on your specific dataset structure
    pass

# Example usage:
if __name__ == "__main__":
    # Initialize predictor
    predictor = CleanlinessPredictor()
    
    # Example prediction
    with open('example_image.jpg', 'rb') as f:
        image_bytes = f.read()
        result = predictor.predict(image_bytes)
        print(result) 