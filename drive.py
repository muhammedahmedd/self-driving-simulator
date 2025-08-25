"""
Autonomous Driving Telemetry Server using SocketIO and Flask.

This script sets up a server that receives real-time telemetry data from a driving simulator,
processes incoming camera images, predicts steering angles using a trained CNN model, and
sends control commands (steering and throttle) back to the simulator.

Author: Mohamed Ahmed
Date: 08/25/2025
"""

# ------------------------------
# Imports
# ------------------------------
import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# ------------------------------
# SocketIO and Flask Setup
# ------------------------------
sio = socketio.Server()
app = Flask(__name__)  # WSGI application

# ------------------------------
# Global Constants
# ------------------------------
speed_limit = 10  # Maximum speed for throttle calculation

# ------------------------------
# Image Preprocessing Function
# ------------------------------
def img_preprocess(img):
    """
    Preprocess simulator image for model input.

    Steps:
    1. Crop image to remove sky and hood (keep road area)
    2. Convert from RGB to YUV color space
    3. Apply Gaussian blur
    4. Resize to model input shape (200x66)
    5. Normalize pixel values to [0, 1]

    Args:
        img (ndarray): Input RGB image.

    Returns:
        ndarray: Preprocessed image suitable for model prediction.
    """
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# ------------------------------
# Telemetry Event Handler
# ------------------------------
@sio.on('telemetry')
def telemetry(sid, data):
    """
    Handle telemetry data sent from the simulator.

    Steps:
    1. Extract speed and image from telemetry.
    2. Preprocess image and feed to model for steering prediction.
    3. Calculate throttle based on current speed and speed limit.
    4. Send control commands (steering and throttle) back to simulator.

    Args:
        sid (str): Session ID of the simulator connection.
        data (dict): Telemetry data containing 'speed' and 'image'.
    """
    speed = float(data['speed'])
    
    # Decode incoming base64 image
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)

    # Preprocess for CNN
    image = img_preprocess(image)
    image = np.array([image])  # Add batch dimension

    # Predict steering angle
    steering_angle = float(model.predict(image))

    # Throttle calculation: reduce throttle at higher speeds
    throttle = 1.0 - speed / speed_limit

    # Debug print
    print('{} {} {}'.format(steering_angle, throttle, speed))

    # Send control command back to simulator
    send_control(steering_angle, throttle)

# ------------------------------
# Connect Event Handler
# ------------------------------
@sio.on('connect')
def connect(sid, environ):
    """
    Handle new simulator connection.

    Args:
        sid (str): Session ID of the simulator connection.
        environ (dict): Environment information.
    """
    print('Connected:', sid)
    send_control(0, 0)  # Initialize with zero steering and throttle

# ------------------------------
# Send Control Function
# ------------------------------
def send_control(steering_angle, throttle):
    """
    Emit steering and throttle commands to the simulator.

    Args:
        steering_angle (float): Predicted steering angle.
        throttle (float): Calculated throttle.
    """
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# ------------------------------
# Server Start
# ------------------------------
if __name__ == '__main__':
    # Load trained model
    model = load_model('model.h5', compile=False)

    # Wrap Flask app with SocketIO WSGI
    app = socketio.WSGIApp(sio, app)

    # Start server on localhost:4567
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 4567)), app)

