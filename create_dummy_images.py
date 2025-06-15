import os
from PIL import Image
import numpy as np

def create_dummy_image(filepath, color):
    img = Image.new('RGB', (10, 10), color)
    img.save(filepath, 'jpeg')

# Ensure directories exist
os.makedirs('training_data/clean', exist_ok=True)
os.makedirs('training_data/dirty', exist_ok=True)

# Create dummy images
create_dummy_image('training_data/clean/dummy_clean_1.jpg', (255, 255, 255)) # White image for clean
create_dummy_image('training_data/dirty/dummy_dirty_1.jpg', (0, 0, 0))   # Black image for dirty

print("Dummy image files created in training_data/clean and training_data/dirty.") 