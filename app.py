from flask import Flask, request, jsonify, render_template
from PIL import Image
import imagehash
import io
import time
import math

app = Flask(__name__)

images_db = []  

HOSPITAL_LAT = 37.4219999
HOSPITAL_LNG = -122.0840575
ALLOWED_RADIUS_METERS = 100 

def haversine(lat1, lng1, lat2, lng2):
    R = 6371000 
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
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

    distance = haversine(lat, lng, HOSPITAL_LAT, HOSPITAL_LNG)
    if distance > ALLOWED_RADIUS_METERS:
        return jsonify({'error': 'Not at hospital location', 'distance_m': distance}), 400

    image_bytes = image_file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img_hash = imagehash.phash(image)
    except Exception:
        return jsonify({'error': 'Invalid image'}), 400

    for entry in images_db:
        if img_hash - entry['hash'] < 5:  
            return jsonify({'error': 'Image too similar to previous upload'}), 400

    images_db.append({
        'hash': img_hash,
        'lat': lat,
        'lng': lng,
        'timestamp': timestamp,
        'image_bytes': image_bytes
    })
    return jsonify({'success': True, 'distance_m': distance})

if __name__ == '__main__':
    app.run(debug=True) 