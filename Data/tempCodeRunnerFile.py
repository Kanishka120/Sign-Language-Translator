import cv2
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
