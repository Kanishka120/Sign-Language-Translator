import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.models import save_model

# Define paths and actions
ISL_DATA_PATH = os.path.join('Data', 'ISL')
BSL_DATA_PATH = os.path.join('Data', 'BSL')
isl_actions = np.array(['Namaste', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning'])
bsl_actions = np.array(['Hello', 'ThankYou', 'I Love You', 'Yes', 'No', 'Good Morning'])

# Load data function
def load_data(data_path, actions):
    features, labels = [], []
    for action in actions:
        action_path = os.path.join(data_path, action)
        for seq in os.listdir(action_path):
            seq_path = os.path.join(action_path, seq)
            for frame in os.listdir(seq_path):
                frame_path = os.path.join(seq_path, frame)
                features.append(np.load(frame_path))
                labels.append(action)
    return np.array(features), np.array(labels)

# Load ISL and BSL data
isl_features, isl_labels = load_data(ISL_DATA_PATH, isl_actions)
bsl_features, bsl_labels = load_data(BSL_DATA_PATH, bsl_actions)

# Combine ISL and BSL data
features = np.concatenate([isl_features, bsl_features])
labels = np.concatenate([isl_labels, bsl_labels])

# Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)  # Encoding both ISL and BSL labels into a single set
labels_categorical = to_categorical(labels_encoded)  # Convert to one-hot encoding

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(features.shape[1],)),  # Input layer
    Dropout(0.2),
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.2),
    Dense(len(np.unique(labels)), activation='softmax')  # Output layer: number of unique labels (ISL + BSL)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Generate predictions for confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)  # Convert one-hot to integer labels

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - ISL to BSL Translation")
plt.show()

# Save the trained model
model.save('translation_model.keras')

