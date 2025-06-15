from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import time
import math
import datetime
from skimage.metrics import structural_similarity as ssim
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import exifread
from numpy.linalg import norm
from cleanliness_model import CleanlinessPredictor
import json
import os
from datetime import datetime

app = Flask(__name__)

IMAGES_DB_FILE = 'images_db.json'

def load_images_db():
    if os.path.exists(IMAGES_DB_FILE):
        try:
            with open(IMAGES_DB_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading images database: {str(e)}")
    return []

def save_images_db(images_db):
    try:
        with open(IMAGES_DB_FILE, 'w') as f:
            json.dump(images_db, f)
    except Exception as e:
        print(f"Error saving images database: {str(e)}")

# Load existing images database
images_db = load_images_db()

HOSPITAL_LAT = 37.4219999
HOSPITAL_LNG = -122.0840575
ALLOWED_RADIUS_METERS = 100 

# --- Rate Limiting Setup ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize the predictor
predictor = CleanlinessPredictor()

def haversine(lat1, lng1, lat2, lng2):
    R = 6371000 
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_image_features(img_bytes):
    """Extract features from image for comparison"""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Resize for consistent comparison
    img = img.resize((256, 256))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Calculate color histogram (3 channels)
    hist_r = np.histogram(img_array[:,:,0], bins=32, range=(0,256))[0]
    hist_g = np.histogram(img_array[:,:,1], bins=32, range=(0,256))[0]
    hist_b = np.histogram(img_array[:,:,2], bins=32, range=(0,256))[0]
    
    # Normalize histograms
    hist_r = hist_r / np.sum(hist_r)
    hist_g = hist_g / np.sum(hist_g)
    hist_b = hist_b / np.sum(hist_b)
    
    # Combine features
    features = np.concatenate([hist_r, hist_g, hist_b])
    
    return features

def image_similarity(img1_bytes, img2_bytes):
    """Compare two images using multiple metrics"""
    # Get features
    features1 = get_image_features(img1_bytes)
    features2 = get_image_features(img2_bytes)
    
    # Calculate histogram similarity (cosine similarity)
    hist_sim = np.dot(features1, features2) / (norm(features1) * norm(features2))
    
    # Calculate structural similarity
    img1 = Image.open(io.BytesIO(img1_bytes)).convert('L').resize((256, 256))
    img2 = Image.open(io.BytesIO(img2_bytes)).convert('L').resize((256, 256))
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    struct_sim = ssim(arr1, arr2)
    
    # Combine similarities (weighted average)
    combined_sim = 0.7 * hist_sim + 0.3 * struct_sim
    
    return combined_sim

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@limiter.limit('5 per minute')
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    try:
        lat = float(request.form['lat'])
        lng = float(request.form['lng'])
        timestamp = float(request.form['timestamp'])
    except Exception:
        return jsonify({'error': 'Invalid or missing lat/lng/timestamp'}), 400

    image_file.seek(0)
    tags = exifread.process_file(image_file, details=False)
    exif_lat = exif_lng = exif_time = None
    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        def dms_to_dd(dms, ref):
            d = float(dms.values[0].num) / float(dms.values[0].den)
            m = float(dms.values[1].num) / float(dms.values[1].den)
            s = float(dms.values[2].num) / float(dms.values[2].den)
            dd = d + m/60 + s/3600
            if ref in ['S', 'W']:
                dd = -dd
            return dd
        exif_lat = dms_to_dd(tags['GPS GPSLatitude'], tags['GPS GPSLatitudeRef'].values)
        exif_lng = dms_to_dd(tags['GPS GPSLongitude'], tags['GPS GPSLongitudeRef'].values)
    if 'EXIF DateTimeOriginal' in tags:
        exif_time_str = str(tags['EXIF DateTimeOriginal'])
        try:
            exif_time_struct = datetime.datetime.strptime(exif_time_str, '%Y:%m:%d %H:%M:%S')
            exif_time = time.mktime(exif_time_struct.timetuple())
        except Exception:
            exif_time = None

    tolerance_deg = 0.0001  
    tolerance_time = 300    
    if exif_lat is not None and exif_lng is not None:
        if abs(exif_lat - lat) > tolerance_deg or abs(exif_lng - lng) > tolerance_deg:
            return jsonify({'error': 'EXIF GPS does not match submitted location', 'exif_lat': exif_lat, 'exif_lng': exif_lng, 'submitted_lat': lat, 'submitted_lng': lng}), 400
    if exif_time is not None:
        if abs(exif_time - timestamp) > tolerance_time:
            return jsonify({'error': 'EXIF timestamp does not match submitted timestamp', 'exif_time': exif_time, 'submitted_timestamp': timestamp}), 400

    image_file.seek(0)
    image_bytes = image_file.read()
    try:
        Image.open(io.BytesIO(image_bytes))
    except Exception:
        return jsonify({'error': 'Invalid image'}), 400

    # Get cleanliness prediction
    cleanliness_result = predictor.predict(image_bytes)
    if 'error' in cleanliness_result:
        return jsonify({'error': f'Cleanliness prediction failed: {cleanliness_result["error"]}'}), 400

    # Image similarity check
    for entry in images_db:
        similarity = image_similarity(image_bytes, entry['image_bytes'])
        if similarity > 0.95:
            return jsonify({'error': 'Image too similar to previous upload', 'similarity': float(similarity)}), 400

    # Create new entry
    new_entry = {
        'lat': lat,
        'lng': lng,
        'timestamp': timestamp,
        'image_bytes': image_bytes,
        'exif_lat': exif_lat,
        'exif_lng': exif_lng,
        'exif_time': exif_time,
        'cleanliness': cleanliness_result,
        'upload_time': datetime.now().isoformat()
    }
    
    # Add to database and save
    images_db.append(new_entry)
    save_images_db(images_db)
    
    return jsonify({
        'success': True,
        'cleanliness': cleanliness_result
    })

if __name__ == '__main__':
    app.run(debug=True) 