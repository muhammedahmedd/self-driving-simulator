"""
Lane Line Detection using OpenCV and Hough Transform.

This script processes a driving video, applies Canny edge detection, masks a region of interest
(lane area), and uses the Hough Transform to detect lane lines. The detected lane lines are
averaged, extrapolated, and displayed on the road in real-time. 

This pipeline is a basic implementation of computer vision for self-driving car applications.

Author: Mohamed Ahmed
Date: 07/10/2025
"""

import cv2 as cv
import numpy as np

# ------------------------------
# Utility Functions
# ------------------------------

def canny(image):
    """
    Apply the Canny Edge Detection pipeline.

    Steps:
        1. Convert image to grayscale.
        2. Apply Gaussian Blur to reduce noise.
        3. Detect edges using Canny.

    Args:
        image (ndarray): Input color image (BGR).

    Returns:
        ndarray: Edge-detected binary image.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    """
    Mask the image to keep only the region of interest (road lanes).

    Args:
        image (ndarray): Edge-detected image.

    Returns:
        ndarray: Masked image with only triangular region of interest.
    """
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])  # Road ROI
    mask = np.zeros_like(image)
    cv.fillPoly(mask, triangle, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


def make_coordinates(image, line_parameters):
    """
    Convert slope and intercept into line endpoints.

    Args:
        image (ndarray): Original frame (used for height).
        line_parameters (tuple): (slope, intercept) of the line.

    Returns:
        ndarray: Coordinates [x1, y1, x2, y2] of the line.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]              # Bottom of the frame
    y2 = int(y1 * (3/5))             # A bit higher (to limit line length)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    """
    Average all detected line segments to form two lane lines.

    Args:
        image (ndarray): Original frame.
        lines (ndarray): Detected Hough line segments.

    Returns:
        ndarray: Two averaged lane lines (left, right).
    """
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Linear fit: slope + intercept
            slope, intercept = parameters
            if slope < 0:  # Left lane slopes negative
                left_fit.append((slope, intercept))
            else:          # Right lane slopes positive
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def display_lines(image, lines):
    """
    Draw lane lines on a blank image.

    Args:
        image (ndarray): Original frame (for shape reference).
        lines (ndarray): Averaged lane line coordinates.

    Returns:
        ndarray: Image with lane lines drawn.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1, y1), (x2, y2), (175, 200, 0), 10)
    return line_image

# ------------------------------
# Video Processing Pipeline
# ------------------------------
cap = cv.VideoCapture('/Users/mohamedahmed/Desktop/test2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Edge detection
    canny_image = canny(frame)

    # 2. Mask region of interest
    cropped_image = region_of_interest(canny_image)

    # 3. Detect lines using Hough Transform
    lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, None, minLineLength=40, maxLineGap=5)

    # 4. Average and extrapolate lane lines
    averaged_lines = average_slope_intercept(frame, lines)

    # 5. Draw lane lines and overlay on original frame
    line_image = display_lines(frame, averaged_lines)
    weighted_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)

    # 6. Display output
    cv.imshow('Lane Lines Detected', weighted_image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# Clean up
# ------------------------------
cap.release()
cv.destroyAllWindows()


   

