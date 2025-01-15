import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic model
mp_holi = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# Define the BSL actions and data collection parameters
bsl_actions = np.array(['Hello', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning'])
no_sequence = 30  # Number of sequences to collect per action
seq_len = 30  # Frames per sequence

# Path for BSL data storage
BSL_DATA_PATH = os.path.join('Data', 'BSL')
os.makedirs(BSL_DATA_PATH, exist_ok=True)

# Ensure directories are created for BSL actions
for action in bsl_actions:
    for seq in range(no_sequence):
        try:
            os.makedirs(os.path.join(BSL_DATA_PATH, action, str(seq)))
        except FileExistsError:
            pass

# Function to detect landmarks with MediaPipe
def mediapipe_detect(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

# Function to extract keypoints
def extract_keypoints(result):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# Data collection for BSL
cap = cv2.VideoCapture(0)
with mp_holi.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in bsl_actions:
        for seq in range(no_sequence):
            for frame_num in range(seq_len):
                ret, frame = cap.read()
                image, result = mediapipe_detect(frame, holistic)

                keypoints = extract_keypoints(result)
                np.save(os.path.join(BSL_DATA_PATH, action, str(seq), f"{frame_num}.npy"), keypoints)

                cv2.putText(image, f'Action: {action}, Sequence: {seq}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('BSL Data Collection', image)

                if cv2.waitKey(50) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
