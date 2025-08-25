"""
German Traffic Sign Recognition using Convolutional Neural Networks (CNNs).

This script loads the German Traffic Sign dataset, visualizes the data distribution, 
preprocesses images (grayscale, histogram equalization, normalization), applies data 
augmentation, and trains a CNN to classify traffic signs. Finally, it demonstrates prediction
on a sample image fetched from the web.

Author: Mohamed Ahmed
Date: 08/2/2025
"""

# ------------------------------
# Imports
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import pickle
import random
import requests
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------
# Set Random Seed for Reproducibility
# ------------------------------
np.random.seed(0)

# ------------------------------
# Load Dataset
# ------------------------------
with open('/Users/mohamedahmed/Documents/german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('/Users/mohamedahmed/Documents/german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('/Users/mohamedahmed/Documents/german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# ------------------------------
# Dataset Assertions
# ------------------------------
assert(X_train.shape[0] == y_train.shape[0]), "Mismatch between X_train and y_train"
assert(X_val.shape[0] == y_val.shape[0]), "Mismatch between X_val and y_val"
assert(X_test.shape[0] == y_test.shape[0]), "Mismatch between X_test and y_test"

assert(X_train.shape[1:] == (32, 32, 3)), "Train images not 32x32x3"
assert(X_val.shape[1:] == (32, 32, 3)), "Validation images not 32x32x3"
assert(X_test.shape[1:] == (32, 32, 3)), "Test images not 32x32x3"

# ------------------------------
# Load Traffic Sign Names
# ------------------------------
data = pd.read_csv("/Users/mohamedahmed/Documents/german-traffic-signs/signnames.csv")
print(data)

# ------------------------------
# Visualize Dataset Distribution
# ------------------------------
num_classes = 43
num_of_samples = []
cols = 5

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1)], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["SignName"])
            num_of_samples.append(len(x_selected))

# Plot class distribution
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# ------------------------------
# Image Preprocessing Functions
# ------------------------------
def grayscale(img):
    """
    Convert color image to grayscale.

    Args:
        img (ndarray): Input BGR image.

    Returns:
        ndarray: Grayscale image.
    """
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def equalize(img):
    """
    Apply histogram equalization to enhance contrast.

    Args:
        img (ndarray): Grayscale image.

    Returns:
        ndarray: Equalized image.
    """
    return cv.equalizeHist(img)

def preprocessing(img):
    """
    Full preprocessing pipeline: grayscale -> equalize -> normalize.

    Args:
        img (ndarray): Original BGR image.

    Returns:
        ndarray: Preprocessed image normalized to [0,1].
    """
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

# Apply preprocessing to all datasets
X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

# Reshape images for CNN input
X_train = X_train.reshape(-1, 32, 32, 1)
X_val = X_val.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)
print("Training data shape:", X_train.shape)

# ------------------------------
# Data Augmentation
# ------------------------------
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10.0
)
datagen.fit(X_train)

# Visualize augmented batch
batches = datagen.flow(X_train, y_train, batch_size=15)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(32, 32), cmap=plt.get_cmap("gray"))
    axs[i].axis("off")
plt.show()
print("Augmented batch shape:", X_batch.shape)

# ------------------------------
# One-Hot Encode Labels
# ------------------------------
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# ------------------------------
# CNN Model Definition
# ------------------------------
def modified_model():
    """
    Define a Convolutional Neural Network for traffic sign classification.

    Returns:
        keras.models.Sequential: Compiled CNN model.
    """
    model = Sequential()
    model.add(Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------
# Train CNN
# ------------------------------
model = modified_model()
print(model.summary())

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=50),
    steps_per_epoch=2000,
    epochs=10,
    validation_data=(X_val, y_val),
    shuffle=True
)

# ------------------------------
# Evaluate Model
# ------------------------------
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# ------------------------------
# Predict on a New Image from Web
# ------------------------------
url = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

# Preprocess image
img = np.asarray(img)
img = cv.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap=plt.get_cmap('gray'))

# Reshape for CNN
img = img.reshape(1, 32, 32, 1)

# Predict
prediction = np.argmax(model.predict(img), axis=1)
print("Predicted sign class:", prediction)
