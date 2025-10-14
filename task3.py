import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# The MFCC features extracted in Task 2 are all saved as .npy files, with a naming convention such as 'name_001.npy'.
# Load Feature
data_dir = "features/"  # Folder for storing MFCC features
X, y = [], []
labels = []  # Save all name categories

for file in os.listdir(data_dir):
    if file.endswith(".npy"):
        name = file.split("_")[0]  # Extract name tags from file names
        if name not in labels:
            labels.append(name)
        label_index = labels.index(name)

        features = np.load(os.path.join(data_dir, file))
        X.append(features)
        y.append(label_index)

X = np.array(X)
y = np.array(y)

# Label One-Hot Encoding
y = to_categorical(y, num_classes=len(labels))

print("Sample number:", X.shape[0], "Categories number:", len(labels))

# Data preprocessing
# Ensure the input shape is (number of samples, height, width, channels)

X = np.expand_dims(X, -1)

# Dividing the training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val)
)

# Assessment and Visualization
plt.plot(history.history['accuracy'], label='Training set accuracy')
plt.plot(history.history['val_accuracy'], label='Validation set accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('DNN train curve')
plt.legend()
plt.show()

#save module
model.save("speech_recogniser_model.h5")
print("Module saved！")
