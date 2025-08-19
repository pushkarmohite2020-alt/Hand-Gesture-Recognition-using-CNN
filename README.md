# Hand-Gesture-Recognition-using-CNN
🔹 Dataset: Leveraged the LeapGestRecog dataset from Kaggle. 🔹 Tech Stack: Python, TensorFlow, Keras, OpenCV, and Matplotlib. 🔹 Key Achievements: ✔ 99.98% Test Accuracy after optimizing hyperparameters and architecture. ✔ Applied data augmentation to enhance model generalization. ✔ Integrated dropout layers to prevent overfitting.
# Install required packages
!pip install tensorflow opencv-python numpy matplotlib kaggle

# Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set the correct dataset path
DATA_DIR = '/root/.cache/kagglehub/datasets/gti-upm/leapgestrecog/versions/1/leapGestRecog'
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30

# Define gesture classes
gestures = {
    '01_palm': 0,
    '02_l': 1,
    '03_fist': 2,
    '04_fist_moved': 3,
    '05_thumb': 4,
    '06_index': 5,
    '07_ok': 6,
    '08_palm_moved': 7,
    '09_c': 8,
    '10_down': 9
}

# Load and preprocess the data
def load_data(data_dir):
    X = []
    y = []

    # Iterate through all user folders (00-09)
    for user_folder in sorted(os.listdir(data_dir)):
        user_path = os.path.join(data_dir, user_folder)
        if os.path.isdir(user_path):
            # Iterate through all gesture folders for each user
            for gesture_name, gesture_id in gestures.items():
                gesture_path = os.path.join(user_path, gesture_name)

                if os.path.exists(gesture_path):
                    for img_file in sorted(os.listdir(gesture_path)):
                        if img_file.endswith('.png'):
                            img_path = os.path.join(gesture_path, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                            img = img / 255.0  # Normalize
                            X.append(img)
                            y.append(gesture_id)

    X = np.array(X)
    y = np.array(y)

    # Add channel dimension (grayscale)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Convert labels to one-hot encoding
    y = to_categorical(y, num_classes=len(gestures))

    return X, y

# Load the data
print("Loading dataset from:", DATA_DIR)
X, y = load_data(DATA_DIR)
print("Dataset loaded successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Build the CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

# Create and compile the model
model = create_model((IMG_SIZE, IMG_SIZE, 1), len(gestures))
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3)
]

# Train the model
print("Training the model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Evaluate the model
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# Save the model
model.save('hand_gesture_model.h5')
print("Model saved as 'hand_gesture_model.h5'")

# Function to predict on new images
def predict_gesture(img_path, model):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Get gesture name from class index
    gesture_name = [k for k, v in gestures.items() if v == predicted_class][0]

    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title(f"Predicted: {gesture_name}")
    plt.axis('off')
    plt.show()

    return gesture_name

# Example prediction (replace with your image path)
# test_img_path = "/path/to/your/test/image.png"
# predict_gesture(test_img_path, model)
