from cleanliness_model import train_model
import os

# Define paths
training_data_path = 'training_data'
model_save_path = os.path.join('models', 'cleanliness_classifier.joblib')

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

print("Starting model training...")
success = train_model(training_data_path, model_save_path)

if success:
    print(f"Model trained successfully and saved to {model_save_path}")
else:
    print("Model training failed.") 