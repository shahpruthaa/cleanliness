# Cleanliness: Hospital Room Cleaning Image Upload App

This is a prototype web app for hospital cleaning staff to upload images of cleaned rooms. The app ensures that:

- Images are captured directly from the device camera (no gallery access)
- The device's current location (latitude/longitude) is attached to each upload
- The image is only accepted if taken within a set radius of the hospital
- Uploaded images are checked for uniqueness using a combination of color histogram and structural similarity
- EXIF metadata is checked for GPS and timestamp consistency
- Uploads are rate-limited to prevent abuse

## Features

- **Camera-only image capture** (no file picker or gallery)
- **Geolocation**: Attaches latitude and longitude to each upload
- **Location validation**: Only allows uploads within 100 meters of the hospital
- **EXIF metadata check**: Ensures image GPS and timestamp match submitted data
- **Image uniqueness**: Uses color histogram and structural similarity to reject images too similar to previous uploads
- **Rate limiting**: 5 uploads per minute per user/IP
- **Simple web interface** for staff

## Setup Instructions

1. **Clone the repository**

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**

   ```bash
   python app.py
   ```

4. **Open the app in your browser**
   - Go to [http://localhost:5000/](http://localhost:5000/) on a device with a camera and geolocation support (e.g., smartphone or laptop)

## Usage

1. Click **Capture Photo** to take a picture using your device's camera.
2. The app will automatically fetch your current location.
3. After capturing, click **Upload** to send the image, timestamp, and location to the server.
4. The server will validate your location, check EXIF metadata, and ensure the image is unique before accepting the upload.

## Notes

- The hospital location is hardcoded in `app.py` (see `HOSPITAL_LAT` and `HOSPITAL_LNG`). Adjust as needed.
- Images and metadata are stored in memory for this prototype. For production, integrate with a database and cloud storage.
- For best results, use a modern browser that supports camera and geolocation APIs.
- The image similarity check is robust and efficient, using both color and structure.

## Next Steps / TODO

- Integrate with Google Cloud Storage for image storage
- Add persistent database (e.g., SQLite, PostgreSQL)
- Add user registration or admin features (optional)
- Admin dashboard for viewing uploads

## Advanced/Future Enhancements

1. **For more robust differentiation**, consider using object detection to identify unique features or layouts in each room, or even QR codes or markers placed in each room for automated recognition.
2. **Nyckel's API** lets you build and deploy custom image classifiers in minutes by uploading labeled samples, after which you can classify new images through a simple API call. The model improves as you provide more data. See [Nyckel's website](https://www.nyckel.com/) for more information.