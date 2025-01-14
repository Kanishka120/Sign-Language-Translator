# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import os

# # Define actions for ISL and BSL
# isl_actions = np.array(['Namaste', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning'])
# bsl_actions = np.array(['Hello', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning'])

# # Load the trained model
# model = load_model('translation_model.keras')

# # Initialize MediaPipe Holistic model
# mp_holi = mp.solutions.holistic
# mp_draw = mp.solutions.drawing_utils

# # Load the LabelEncoder from the training process
# encoder = LabelEncoder()
# encoder.fit(np.concatenate([isl_actions, bsl_actions]))

# # Function to extract keypoints (258 features)
# def extract_keypoints(result):
#     # Extract pose keypoints (x, y, z, visibility) - 33 landmarks * 4 = 132 features
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark[:33]]).flatten() if result.pose_landmarks else np.zeros(33 * 4)
    
#     # Extract face keypoints (x, y, z) - 10 landmarks * 3 = 30 features (this is adjusted)
#     # 10 face landmarks should suffice for reducing to 258 features (less than 68)
#     face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark[:20]]).flatten() if result.face_landmarks else np.zeros(20 * 3)
    
#     # Extract left hand keypoints (x, y, z) - 6 landmarks * 3 = 18 features (adjusted to match total 258 features)
#     lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark[:21]]).flatten() if result.left_hand_landmarks else np.zeros(21 * 3)
    
#     # Extract right hand keypoints (x, y, z) - 6 landmarks * 3 = 18 features (adjusted to match total 258 features)
#     rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark[:21]]).flatten() if result.right_hand_landmarks else np.zeros(21 * 3)
    
#     # Return the concatenated feature array (this will now be 258 features)
#     return np.concatenate([pose, face, lh, rh])  # Total = 258 features

# # Real-time video capture and gesture prediction
# cap = cv2.VideoCapture(0)

# with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue
        
#         # Process frame with MediaPipe
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         image.flags.writeable = False  # Make image not writable for efficiency
#         result = holistic.process(image)  # Process the image to get landmarks
#         image.flags.writeable = True  # Make image writable again
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR

#         # Extract keypoints
#         keypoints = extract_keypoints(result)

#         # Reshape the keypoints into the required shape for the model
#         keypoints = keypoints.reshape(1, -1)  # Flatten the keypoints to match input shape

#         # Predict the BSL gesture from the ISL gesture
#         prediction = model.predict(keypoints)
#         predicted_class = np.argmax(prediction, axis=1)
#         predicted_bsl_action = bsl_actions[predicted_class[0]]

#         # Display the predicted BSL action on the screen
#         cv2.putText(image, f'Predicted Action: {predicted_bsl_action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         # Draw landmarks on the image
#         if result.face_landmarks:
#             mp_draw.draw_landmarks(image, result.face_landmarks, mp_holi.FACEMESH_CONTOURS,
#                                    mp_draw.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
#                                    mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
#         if result.pose_landmarks:
#             mp_draw.draw_landmarks(image, result.pose_landmarks, mp_holi.POSE_CONNECTIONS,
#                                    mp_draw.DrawingSpec(color=(80, 55, 40), thickness=2, circle_radius=4),
#                                    mp_draw.DrawingSpec(color=((255, 77, 77)), thickness=2, circle_radius=2))
#         if result.left_hand_landmarks:
#             mp_draw.draw_landmarks(image, result.left_hand_landmarks, mp_holi.HAND_CONNECTIONS,
#                                    mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                                    mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
#         if result.right_hand_landmarks:
#             mp_draw.draw_landmarks(image, result.right_hand_landmarks, mp_holi.HAND_CONNECTIONS,
#                                    mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                                    mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

#         # Show the real-time video with gesture prediction
#         cv2.imshow('Real-Time ISL to BSL Gesture Translation', image)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

# # Define actions for ISL and BSL
# isl_actions = ['Namaste', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']
# bsl_actions = ['Hello', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']

# # ISL to BSL mapping (this is just an example, adjust the mapping as per actual correspondence)
# isl_to_bsl_mapping = {
#     'Namaste': 'Hello',
#     'ThankYou': 'ThankYou',
#     'I Love You': 'I Love You',
#     'Yes': 'Yes',
#     'No': 'No',
#     'Good Morning': 'Good Morning'
# }

# # Load the trained model and LabelEncoder
# model = load_model('translation_model.keras')
# encoder = LabelEncoder()
# encoder.fit(np.concatenate([isl_actions, bsl_actions]))  # Combining both actions for encoding

# # Initialize MediaPipe Holistic model
# mp_holi = mp.solutions.holistic
# mp_draw = mp.solutions.drawing_utils

# # Function to extract keypoints (258 features)
# # def extract_keypoints(result):
# #     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark[:33]]).flatten() if result.pose_landmarks else np.zeros(33 * 4)
# #     face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark[:20]]).flatten() if result.face_landmarks else np.zeros(20 * 3)
# #     lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark[:21]]).flatten() if result.left_hand_landmarks else np.zeros(21 * 3)
# #     rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark[:21]]).flatten() if result.right_hand_landmarks else np.zeros(21 * 3)
# #     return np.concatenate([pose, face, lh, rh])

# def extract_keypoints(result):
#     # Pose: 33 landmarks * 4 features = 132
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark[:33]]).flatten() if result.pose_landmarks else np.zeros(33 * 4)

#     # Face: 20 landmarks * 3 features = 60
#     face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark[:20]]).flatten() if result.face_landmarks else np.zeros(20 * 3)

#     # Left Hand: 21 landmarks * 3 features = 63
#     lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark[:21]]).flatten() if result.left_hand_landmarks else np.zeros(21 * 3)

#     # Right Hand: 21 landmarks * 3 features = 63
#     rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark[:21]]).flatten() if result.right_hand_landmarks else np.zeros(21 * 3)

#     # If the total number of features exceeds 258, truncate to match the expected shape
#     total_features = np.concatenate([pose, face, lh, rh])

#     # Truncate to 258 features
#     return total_features[:258]


# # Real-time video capture and gesture prediction
# cap = cv2.VideoCapture(0)

# with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue
        
#         # Process frame with MediaPipe
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = holistic.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Extract keypoints
#         keypoints = extract_keypoints(result).reshape(1, -1)

#         # Predict the ISL gesture first (assuming model is trained on ISL gestures)
#         prediction = model.predict(keypoints)
#         predicted_class = np.argmax(prediction, axis=1)
#         predicted_isl_action = isl_actions[predicted_class[0]]

#         # Map ISL action to BSL action
#         predicted_bsl_action = isl_to_bsl_mapping.get(predicted_isl_action, 'Unknown')

#         # Display the predicted BSL action
#         cv2.putText(image, f'Predicted BSL: {predicted_bsl_action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Draw landmarks
#         if result.face_landmarks: mp_draw.draw_landmarks(image, result.face_landmarks, mp_holi.FACEMESH_CONTOURS)
#         if result.pose_landmarks: mp_draw.draw_landmarks(image, result.pose_landmarks, mp_holi.POSE_CONNECTIONS)
#         if result.left_hand_landmarks: mp_draw.draw_landmarks(image, result.left_hand_landmarks, mp_holi.HAND_CONNECTIONS)
#         if result.right_hand_landmarks: mp_draw.draw_landmarks(image, result.right_hand_landmarks, mp_holi.HAND_CONNECTIONS)

#         # Show the real-time video with gesture prediction
#         cv2.imshow('ISL to BSL Gesture Translation', image)

#         # Exit on 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

# # Define actions for ISL and BSL
# isl_actions = ['Namaste', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']
# bsl_actions = ['Hello', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']

# # ISL to BSL mapping (this is just an example, adjust the mapping as per actual correspondence)
# isl_to_bsl_mapping = {
#     'Namaste': 'Hello',
#     'ThankYou': 'ThankYou',
#     'I Love You': 'I Love You',
#     'Yes': 'Yes',
#     'No': 'No',
#     'Good Morning': 'Good Morning'
# }

# # Load the trained model and LabelEncoder
# model = load_model('translation_model.keras')
# encoder = LabelEncoder()
# encoder.fit(np.concatenate([isl_actions, bsl_actions]))  # Combining both actions for encoding

# # Initialize MediaPipe Holistic model
# mp_holi = mp.solutions.holistic
# mp_draw = mp.solutions.drawing_utils

# # Function to extract keypoints (258 features)
# def extract_keypoints(result):
#     # Pose: 33 landmarks * 4 features = 132
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark[:33]]).flatten() if result.pose_landmarks else np.zeros(33 * 4)

#     # Face: 20 landmarks * 3 features = 60
#     face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark[:20]]).flatten() if result.face_landmarks else np.zeros(20 * 3)

#     # Left Hand: 21 landmarks * 3 features = 63
#     lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark[:21]]).flatten() if result.left_hand_landmarks else np.zeros(21 * 3)

#     # Right Hand: 21 landmarks * 3 features = 63
#     rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark[:21]]).flatten() if result.right_hand_landmarks else np.zeros(21 * 3)

#     # If the total number of features exceeds 258, truncate to match the expected shape
#     total_features = np.concatenate([pose, face, lh, rh])

#     # Truncate to 258 features
#     return total_features[:258]


# # Real-time video capture and gesture prediction
# cap = cv2.VideoCapture(0)

# with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue
        
#         # Process frame with MediaPipe
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = holistic.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Extract keypoints
#         keypoints = extract_keypoints(result).reshape(1, -1)

#         # Predict the ISL gesture first (assuming model is trained on ISL gestures)
#         prediction = model.predict(keypoints)

#         # Handle the case where the model's prediction is invalid
#         if prediction is not None and len(prediction) > 0:
#             predicted_class = np.argmax(prediction, axis=1)
#             if predicted_class[0] < len(isl_actions):  # Check if predicted class is within range
#                 predicted_isl_action = isl_actions[predicted_class[0]]

#                 # Map ISL action to BSL action
#                 predicted_bsl_action = isl_to_bsl_mapping.get(predicted_isl_action, 'Unknown')
#             else:
#                 predicted_isl_action = 'Unknown'
#                 predicted_bsl_action = 'Unknown'
#         else:
#             predicted_isl_action = 'Unknown'
#             predicted_bsl_action = 'Unknown'

#         # Display the predicted BSL action
#         cv2.putText(image, f'Predicted BSL: {predicted_bsl_action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Draw landmarks
#         if result.face_landmarks: mp_draw.draw_landmarks(image, result.face_landmarks, mp_holi.FACEMESH_CONTOURS)
#         if result.pose_landmarks: mp_draw.draw_landmarks(image, result.pose_landmarks, mp_holi.POSE_CONNECTIONS)
#         if result.left_hand_landmarks: mp_draw.draw_landmarks(image, result.left_hand_landmarks, mp_holi.HAND_CONNECTIONS)
#         if result.right_hand_landmarks: mp_draw.draw_landmarks(image, result.right_hand_landmarks, mp_holi.HAND_CONNECTIONS)

#         # Show the real-time video with gesture prediction
#         cv2.imshow('ISL to BSL Gesture Translation', image)

#         # Exit on 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

# # Define actions for ISL and BSL
# isl_actions = ['Namaste', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']
# bsl_actions = ['Hello', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']

# # ISL to BSL mapping (this is just an example, adjust the mapping as per actual correspondence)
# isl_to_bsl_mapping = {
#     'Namaste': 'Hello',
#     'ThankYou': 'ThankYou',
#     'I Love You': 'I Love You',
#     'Yes': 'Yes',
#     'No': 'No',
#     'Good Morning': 'Good Morning'
# }

# # Load the trained model and LabelEncoder
# model = load_model('translation_model.keras')
# encoder = LabelEncoder()
# encoder.fit(np.concatenate([isl_actions, bsl_actions]))  # Combining both actions for encoding

# # Initialize MediaPipe Holistic model
# mp_holi = mp.solutions.holistic
# mp_draw = mp.solutions.drawing_utils

# # Function to extract keypoints (258 features)
# def extract_keypoints(result):
#     # Pose: 33 landmarks * 4 features = 132
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark[:33]]).flatten() if result.pose_landmarks else np.zeros(33 * 4)

#     # Face: 20 landmarks * 3 features = 60
#     face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark[:20]]).flatten() if result.face_landmarks else np.zeros(20 * 3)

#     # Left Hand: 21 landmarks * 3 features = 63
#     lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark[:21]]).flatten() if result.left_hand_landmarks else np.zeros(21 * 3)

#     # Right Hand: 21 landmarks * 3 features = 63
#     rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark[:21]]).flatten() if result.right_hand_landmarks else np.zeros(21 * 3)

#     # If the total number of features exceeds 258, truncate to match the expected shape
#     total_features = np.concatenate([pose, face, lh, rh])

#     # Truncate to 258 features
#     return total_features[:258]


# # Real-time video capture and gesture prediction
# cap = cv2.VideoCapture(0)

# with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue
        
#         # Process frame with MediaPipe
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = holistic.process(image)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Extract keypoints
#         keypoints = extract_keypoints(result).reshape(1, -1)

#         # Predict the ISL gesture first (assuming model is trained on ISL gestures)
#         prediction = model.predict(keypoints)

#         # Handle the case where the model's prediction is invalid
#         if prediction is not None and len(prediction) > 0:
#             predicted_class = np.argmax(prediction, axis=1)
#             if predicted_class[0] < len(isl_actions):  # Check if predicted class is within range
#                 predicted_isl_action = isl_actions[predicted_class[0]]

#                 # Map ISL action to BSL action
#                 predicted_bsl_action = isl_to_bsl_mapping.get(predicted_isl_action, 'Unknown')
#             else:
#                 predicted_isl_action = 'Unknown'
#                 predicted_bsl_action = 'Unknown'
#         else:
#             predicted_isl_action = 'Unknown'
#             predicted_bsl_action = 'Unknown'

#         # Display the predicted BSL action
#         cv2.putText(image, f'Predicted BSL: {predicted_bsl_action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Draw landmarks
#         if result.face_landmarks: mp_draw.draw_landmarks(image, result.face_landmarks, mp_holi.FACEMESH_CONTOURS)
#         if result.pose_landmarks: mp_draw.draw_landmarks(image, result.pose_landmarks, mp_holi.POSE_CONNECTIONS)
#         if result.left_hand_landmarks: mp_draw.draw_landmarks(image, result.left_hand_landmarks, mp_holi.HAND_CONNECTIONS)
#         if result.right_hand_landmarks: mp_draw.draw_landmarks(image, result.right_hand_landmarks, mp_holi.HAND_CONNECTIONS)

#         # Show the real-time video with gesture prediction
#         cv2.imshow('ISL to BSL Gesture Translation', image)

#         # Exit on 'q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Define actions for ISL and BSL
isl_actions = ['Namaste', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']
bsl_actions = ['Hello', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning']

# ISL to BSL mapping (this is just an example, adjust the mapping as per actual correspondence)
isl_to_bsl_mapping = {
    'Namaste': 'Hello',
    'ThankYou': 'ThankYou',
    'I Love You': 'I Love You',
    'Yes': 'Yes',
    'No': 'No',
    'Good Morning': 'Good Morning'
}

# Load the trained model and LabelEncoder
model = load_model('translation_model.keras')
encoder = LabelEncoder()
encoder.fit(np.concatenate([isl_actions, bsl_actions]))  # Combining both actions for encoding

# Initialize MediaPipe Holistic model
mp_holi = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# Function to extract keypoints (258 features)
def extract_keypoints(result):
    # Pose: 33 landmarks * 4 features = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark[:33]]).flatten() if result.pose_landmarks else np.zeros(33 * 4)

    # Face: 20 landmarks * 3 features = 60
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark[:20]]).flatten() if result.face_landmarks else np.zeros(20 * 3)

    # Left Hand: 21 landmarks * 3 features = 63
    lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark[:21]]).flatten() if result.left_hand_landmarks else np.zeros(21 * 3)

    # Right Hand: 21 landmarks * 3 features = 63
    rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark[:21]]).flatten() if result.right_hand_landmarks else np.zeros(21 * 3)

    # If the total number of features exceeds 258, truncate to match the expected shape
    total_features = np.concatenate([pose, face, lh, rh])

    # Truncate to 258 features
    return total_features[:258]


# Real-time video capture and gesture prediction
cap = cv2.VideoCapture(0)

with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process frame with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract keypoints
        keypoints = extract_keypoints(result).reshape(1, -1)

        # Predict the ISL gesture first (assuming model is trained on ISL gestures)
        prediction = model.predict(keypoints)

        # Handle the case where the model's prediction is invalid
        if prediction is not None and len(prediction) > 0:
            predicted_class = np.argmax(prediction, axis=1)
            if predicted_class[0] < len(isl_actions):  # Check if predicted class is within range
                predicted_isl_action = isl_actions[predicted_class[0]]

                # Map ISL action to BSL action
                predicted_bsl_action = isl_to_bsl_mapping.get(predicted_isl_action, 'Unknown')
            else:
                predicted_isl_action = 'Unknown'
                predicted_bsl_action = 'Unknown'
        else:
            predicted_isl_action = 'Unknown'
            predicted_bsl_action = 'Unknown'

        # Display the predicted BSL action
        cv2.putText(image, f'Predicted BSL: {predicted_bsl_action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks
        if result.face_landmarks: mp_draw.draw_landmarks(image, result.face_landmarks, mp_holi.FACEMESH_CONTOURS)
        if result.pose_landmarks: mp_draw.draw_landmarks(image, result.pose_landmarks, mp_holi.POSE_CONNECTIONS)
        if result.left_hand_landmarks: mp_draw.draw_landmarks(image, result.left_hand_landmarks, mp_holi.HAND_CONNECTIONS)
        if result.right_hand_landmarks: mp_draw.draw_landmarks(image, result.right_hand_landmarks, mp_holi.HAND_CONNECTIONS)

        # Show the real-time video with gesture prediction
        cv2.imshow('ISL to BSL Gesture Translation', image)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
