import streamlit as st
import cv2
import numpy as np
from io import BytesIO

def read_image(file_bytes):
    image = np.frombuffer(file_bytes, dtype=np.uint8)
    return image.reshape((256, 256))

# If you want to read and save the processed images to binary files
# def write_image(file_path, image):
#     with open(file_path, 'wb') as file:
#         image.tofile(file)

def apply_median_filter(image):
    return cv2.medianBlur(image, ksize=3)

def apply_morphological_opening(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def apply_morphological_closing(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

st.header("Image Processing with Streamlit")

# Upload images
uploaded_file9 = st.file_uploader("Choose an image for camera9", type=["bin"])
uploaded_file99 = st.file_uploader("Choose an image for camera99", type=["bin"])

if uploaded_file9 and uploaded_file99:
    # Read images
    image9 = read_image(uploaded_file9.read())
    image99 = read_image(uploaded_file99.read())

    # Apply filters camera9.bin
    filtered_image9 = apply_median_filter(image9)
    opened_image9 = apply_morphological_opening(image9)
    closed_image9 = apply_morphological_closing(image9)

    # Apply filters camera99.bin
    filtered_image99 = apply_median_filter(image99)
    opened_image99 = apply_morphological_opening(image99)
    closed_image99 = apply_morphological_closing(image99)

    # camera9.bin
    col1, col2, col3, col4 = st.columns(4)

    col1.image(image9, caption="Original Camera 9", use_column_width=True)
    col2.image(filtered_image9, caption="Median Filter", use_column_width=True)
    col3.image(opened_image9, caption="Morphological Opening", use_column_width=True)
    col4.image(closed_image9, caption="Morphological Closing", use_column_width=True)

    # camera99.bin
    col1, col2, col3, col4 = st.columns(4)

    col1.image(image99, caption="Original Camera 99", use_column_width=True)
    col2.image(filtered_image99, caption="Median Filter", use_column_width=True)
    col3.image(opened_image99, caption="Morphological Opening", use_column_width=True)
    col4.image(closed_image99, caption="Morphological Closing", use_column_width=True)

    # If you want to save the processed images to binary files
    #
    # # Save images for camera 9
    # write_image('filtered_camera9.bin', filtered_image9)
    # write_image('opened_camera9.bin', opened_image9)
    # write_image('closed_camera9.bin', closed_image9)
    #
    # # Save images for camera 99
    # write_image('filtered_camera99.bin', filtered_image99)
    # write_image('opened_camera99.bin', opened_image99)
    # write_image('closed_camera99.bin', closed_image99)
