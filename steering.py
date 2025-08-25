"""
Autonomous Steering Angle Prediction using NVIDIA CNN Architecture.

This script loads driving images and steering angle data, balances the dataset,
applies data augmentation (zoom, pan, brightness, flip), preprocesses images,
and trains a CNN model based on the NVIDIA architecture to predict steering angles
for autonomous driving. Real-time batch generators are used for efficient training.

Author: Mohamed Ahmed
Date: 08/25/2025
"""

# ------------------------------
# Imports
# ------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import pandas as pd
import random
import ntpath

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

# ------------------------------
# Set Random Seeds for Reproducibility
# ------------------------------
np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

# ------------------------------
# Dataset Paths
# ------------------------------
datadir = "/Users/mohamedahmed/Documents/steering data"
csv_file = os.path.join(datadir, "driving_log.csv")

columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(csv_file, names=columns)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Original data sample:")
print(data.head())

# ------------------------------
# Extract Filenames from Full Paths
# ------------------------------
def path_leaf(path):
    """
    Extract the filename from a full file path.

    Args:
        path (str): Full file path.

    Returns:
        str: Filename.
    """
    head, tail = ntpath.split(path)
    return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

print("\nAfter path cleaning:")
print(data.head())

# ------------------------------
# Balance Dataset
# ------------------------------
num_bins = 25
samples_per_bin = 400

hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5

# Visualize before balancing
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.title("Before balancing")
plt.show()

remove_list = []
for j in range(num_bins):
    bin_indices = data[(data['steering'] >= bins[j]) & (data['steering'] <= bins[j+1])].index
    bin_indices = shuffle(bin_indices)
    remove_list.extend(bin_indices[samples_per_bin:])  # keep only first N

data.drop(remove_list, inplace=True)

print('Removed:', len(remove_list))
print('Remaining:', len(data))

# Visualize after balancing
hist, _ = np.histogram(data['steering'], num_bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.title("After balancing")
plt.show()

# ------------------------------
# Load Image Paths and Steering Angles
# ------------------------------
def load_img_steering(img_dir, df, correction=0.15):
    """
    Load image paths and steering angles from CSV data. Applies correction for left/right cameras.

    Args:
        img_dir (str): Directory containing images.
        df (DataFrame): Driving log DataFrame.
        correction (float): Steering angle adjustment for side cameras.

    Returns:
        tuple: (image_paths, steering_angles) as numpy arrays.
    """
    image_paths = []
    steerings = []

    for i in range(len(df)):
        row = df.iloc[i]

        # Center image
        image_paths.append(os.path.join(img_dir, row['center'].strip()))
        steerings.append(float(row['steering']))

        # Left image with correction
        image_paths.append(os.path.join(img_dir, row['left'].strip()))
        steerings.append(float(row['steering']) + correction)

        # Right image with correction
        image_paths.append(os.path.join(img_dir, row['right'].strip()))
        steerings.append(float(row['steering']) - correction)

    return np.asarray(image_paths), np.asarray(steerings)

image_paths, steerings = load_img_steering(os.path.join(datadir, "IMG"), data)

print("\nDataset loaded:")
print("Total images:", len(image_paths))
print("Total steerings:", len(steerings))
print("Example image path:", image_paths[0])
print("Example steering angle:", steerings[0])

# ------------------------------
# Train/Validation Split
# ------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    image_paths, steerings, test_size=0.2, random_state=6
)

print("\nTraining Samples: {}\nValidation Samples: {}".format(len(X_train), len(X_valid)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')
plt.show()

# ------------------------------
# Data Augmentation Functions
# ------------------------------
def zoom(image):
    """
    Randomly zoom the image.

    Args:
        image (ndarray): Input image.

    Returns:
        ndarray: Zoomed image.
    """
    zoom_aug = iaa.Affine(scale=(1, 1.3))
    return zoom_aug.augment_image(image)

def pan(image):
    """
    Randomly pan the image in x/y directions.

    Args:
        image (ndarray): Input image.

    Returns:
        ndarray: Panned image.
    """
    pan_aug = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return pan_aug.augment_image(image)

def img_random_brightness(image):
    """
    Randomly adjust image brightness.

    Args:
        image (ndarray): Input image.

    Returns:
        ndarray: Brightness-adjusted image.
    """
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(image)

def img_random_flip(image, steering_angle):
    """
    Horizontally flip image and invert steering angle.

    Args:
        image (ndarray): Input image.
        steering_angle (float): Original steering angle.

    Returns:
        tuple: (flipped_image, flipped_steering_angle)
    """
    image = cv.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augment(image_path, steering_angle):
    """
    Apply random augmentation: pan, zoom, brightness, flip.

    Args:
        image_path (str): Path to input image.
        steering_angle (float): Steering angle.

    Returns:
        tuple: (augmented_image, augmented_steering_angle)
    """
    image = mpimg.imread(image_path)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)
    return image, steering_angle

# ------------------------------
# Preprocessing Function
# ------------------------------
def img_preprocess(img):
    """
    Crop, convert to YUV, blur, resize, normalize image.

    Args:
        img (ndarray): Input image.

    Returns:
        ndarray: Preprocessed image.
    """
    img = img[60:135, :, :]
    img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.resize(img, (200, 66))
    img = img / 255.0
    return img

# ------------------------------
# Batch Generator
# ------------------------------
def batch_generator(image_paths, steering_angles, batch_size, is_training):
    """
    Generate batches of images and steering angles for training/validation.

    Args:
        image_paths (ndarray): Paths to images.
        steering_angles (ndarray): Steering angles.
        batch_size (int): Batch size.
        is_training (bool): Apply augmentation if True.

    Yields:
        tuple: (batch_images, batch_steering_angles)
    """
    while True:
        batch_img = []
        batch_steering = []

        for _ in range(batch_size):
            idx = random.randint(0, len(image_paths) - 1)

            if is_training:
                img, steering = random_augment(image_paths[idx], steering_angles[idx])
            else:
                img = mpimg.imread(image_paths[idx])
                steering = steering_angles[idx]

            img = img_preprocess(img)
            batch_img.append(img)
            batch_steering.append(steering)

        yield (np.asarray(batch_img), np.asarray(batch_steering))

# ------------------------------
# NVIDIA CNN Model
# ------------------------------
def nvidia_model():
    """
    Define the NVIDIA CNN architecture for end-to-end steering angle prediction.

    Returns:
        keras.models.Sequential: Compiled CNN model.
    """
    model = Sequential()
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu', input_shape=(66,200,3)))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=optimizer)
    return model

# ------------------------------
# Train Model
# ------------------------------
model = nvidia_model()
print(model.summary())

history = model.fit(
    batch_generator(X_train, y_train, 100, True),
    steps_per_epoch=300,
    epochs=10,
    validation_data=batch_generator(X_valid, y_valid, 100, False),
    validation_steps=200,
    verbose=1,
    shuffle=True
)

# ------------------------------
# Plot Training History
# ------------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

# ------------------------------
# Save Model
# ------------------------------
model.save('model.h5')






