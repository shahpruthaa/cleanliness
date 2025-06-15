# Cleanliness: Hospital Room Cleaning Image Upload App

This is a prototype web app designed for hospital cleaning staff to streamline the process of uploading images of cleaned rooms. The application leverages various technologies to ensure data integrity, validate image context, and provide cleanliness predictions.

## How it Works

1.  **Image Capture**: Users capture photos directly from their device's camera through the web interface. Gallery access is disabled to ensure real-time image capture.
2.  **Geolocation & Timestamp**: The app automatically retrieves the device's current latitude, longitude, and a timestamp at the time of capture. This data is submitted along with the image.
3.  **Backend Validation (Flask `app.py`)**: Upon receiving an upload, the Flask backend performs several crucial validations:
    - **Location Validation**: It checks if the uploaded image's location is within a predefined radius of the hospital's coordinates.
    - **EXIF Metadata Consistency**: If available, EXIF GPS and timestamp data embedded in the image are compared against the submitted location and timestamp for consistency.
    - **Image Uniqueness**: To prevent duplicate uploads, a similarity check is performed against previously uploaded images using a combination of color histogram and structural similarity. Images that are too similar are rejected.
    - **Rate Limiting**: Uploads are rate-limited per user/IP to prevent abuse.
4.  **Cleanliness Prediction (Python `cleanliness_model.py`)**: After initial validations, the image is passed to a trained machine learning model to predict its cleanliness state. The model outputs a classification (e.g., 'Clean', 'Messy') and a confidence score.
5.  **Data Storage**: Image metadata and cleanliness predictions are stored in a local JSON file (`images_db.json`). Raw image bytes are Base64 encoded for JSON serialization.
6.  **Frontend Feedback**: The results of the upload and cleanliness prediction are displayed to the user on the web interface.

## Technologies Used

### Backend (Python Flask)

- **Flask**: Web framework for building the API endpoints.
- **Pillow (PIL)**: Image processing library for opening, converting, and resizing images.
- **NumPy**: Fundamental package for numerical computation, used in image feature extraction and similarity calculations.
- **scikit-image**: Used for advanced image processing, specifically for calculating Structural Similarity Index (SSIM).
- **scikit-learn**: Machine learning library used to build and train the `RandomForestClassifier` for cleanliness prediction.
- **Joblib**: Used for efficient saving and loading of the trained machine learning model.
- **Flask-Limiter**: For implementing request rate limiting.
- **ExifRead**: To extract EXIF metadata (GPS, timestamp) from uploaded images.

### Frontend (HTML, CSS, JavaScript)

- **HTML5**: Structure of the web application.
- **CSS3**: Styling of the user interface.
- **JavaScript**: Handles camera access, geolocation, image capture, form data submission via AJAX, and dynamic updating of the UI.
  - Utilizes `navigator.mediaDevices.getUserMedia` for camera access.
  - Leverages `navigator.geolocation.getCurrentPosition` for location services.

## Features

- **Camera-only image capture** (no file picker or gallery)
- **Geolocation**: Attaches latitude and longitude to each upload
- **Location validation**: Only allows uploads within 100 meters of the hospital
- **EXIF metadata check**: Ensures image GPS and timestamp match submitted data
- **Image uniqueness**: Uses color histogram and structural similarity to reject images too similar to previous uploads
- **Rate limiting**: 5 uploads per minute per user/IP
- **Cleanliness Prediction**: Classifies images as 'Clean', 'Needs Attention', or 'Dirty' using a trained machine learning model.
- **Simple web interface** for staff

## Cleanliness Prediction Model

This application incorporates a machine learning model (`cleanliness_model.py`) to predict the cleanliness of uploaded images. The model uses image features such as brightness, contrast, and edge intensity to classify rooms. It is a `RandomForestClassifier` trained on labeled data.

- The trained model is saved in the `models/` directory as `cleanliness_classifier.joblib`.
- The model can be updated based on labeled clean and dirty images to refine its prediction accuracy. To train the model, organize your images in `training_data/clean` and `training_data/dirty` and run `python run_training.py`.

## Setup Instructions

1.  **Clone the repository**

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Training Data (Optional but Recommended for Accuracy)**
    Create a `training_data` directory in the project root. Inside it, create `clean` and `dirty` subdirectories. Place your corresponding images in these folders. For example:

    ```
    cleanliness/
    ├── training_data/
    │   ├── clean/
    │   │   ├── clean_room_1.jpg
    │   │   └── ...
    │   └── dirty/
    │       ├── dirty_room_1.jpg
    │       └── ...
    └── ...
    ```

4.  **Train the Cleanliness Model (Optional but Recommended)**
    If you've added training data, run the training script:

    ```bash
    python run_training.py
    ```

5.  **Run the Flask app**

    ```bash
    python app.py
    ```

6.  **Open the app in your browser**
    - Go to [http://localhost:5000/](http://localhost:5000/) on a device with a camera and geolocation support (e.g., smartphone or laptop)

## Usage

1.  Click **Capture Photo** to take a picture using your device's camera.
2.  The app will automatically fetch your current location.
3.  After capturing, click **Upload** to send the image, timestamp, and location to the server.
4.  The server will validate your location, check EXIF metadata, perform image uniqueness checks, and run the cleanliness prediction. The result will be displayed on the page.

## Notes

- The hospital location is hardcoded in `app.py` (see `HOSPITAL_LAT` and `HOSPITAL_LNG`). Adjust as needed.
- Images and metadata are stored in a local JSON file (`images_db.json`) for this prototype. For production, integrate with a robust database and cloud storage solution.
- For best results, use a modern browser that supports camera and geolocation APIs.
- The image similarity check uses both color histogram and structural similarity for robustness.
- The accuracy of the cleanliness prediction model heavily depends on the quality and quantity of the training data provided.

## Next Steps / TODO

- Integrate with Google Cloud Storage for image storage
- Add persistent database (e.g., SQLite, PostgreSQL)
- Add user registration or admin features (optional)
- Admin dashboard for viewing uploads

## Advanced/Future Enhancements

1.  **For more robust differentiation**, consider using object detection to identify unique features or layouts in each room, or even QR codes or markers placed in each room for automated recognition.
2.  **Nyckel's API** lets you build and deploy custom image classifiers in minutes by uploading labeled samples, after which you can classify new images through a simple API call. The model improves as you provide more data. See [Nyckel's website](https://www.nyckel.com/) for more information.
