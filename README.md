# Indian Sign Language to British Sign Language Translator

## Overview
The ISL to BSL Gesture Translator is a real-time system designed to interpret gestures in Indian Sign Language (ISL) and translate them into their corresponding British Sign Language (BSL) gestures. Using machine learning and computer vision, this project promotes inclusivity and enhances communication for sign language users across different linguistic regions.

## File Overview
### 1. ISL Data Collection(datacollection_isl.py)
- Uses a webcam to record isl gestures and extract pose, hand, and face keypoints.
- Saves the keypoint data as .npy files.
### 2. BSL Data Collection(datacollection_bsl.py)
- Uses a webcam to record bsl gestures and extract pose, hand, and face keypoints.
- Saves the keypoint data as .npy files.
### 3. Training of Model(train_model.py)
- Combines ISL and BSL data, encodes gestures, and builds a classification model.
- Saves the trained model for real-time gesture recognition.
- Supports the addition of new gestures by expanding the dataset.
### 4. Model Testing(test.py)
- Recognizes ISL gestures and maps them to their corresponding BSL gestures.
- Displays the predicted BSL gesture text on the video feed.

## Features
- **Real Time Gesture Recognition :** Detects ISL gestures in real time using webcam.
- **ISL to BSL mapping :** Each ISL gesture is translated to its equivalent gesture in BSL.
- **On-screen Result :** The translation is shown as text on the video feed.
- **Pre Trained Model :** Uses a neural network trained on specific datasets for recognition.
- **Extenibility :** Extendable to include more gestures and other sign languages.

## Working
- Captures gestures through a webcam
- Uses MediaPipe Holistic to extract pose, face, and hand landmarks
- Identification of ISL gestures by pre trained neural network
- ISL gestures are mapped to corresponding BSL gestures using a predefined dictionary
- BSL translation is shown on videofeed

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Keras
- NumPy
- Matplotlib
 
## Future Work
- Add support for more gestures and additional sign languages
- Improve model accuracy with larger and more diverse datasets
- Implentation of audio feedback for BSL translation

## Acknowledgement
- MediaPipe by Google for keypoint extraction.
- Keras and TensorFlow for model development.
- OpenCV for real-time video processing.
