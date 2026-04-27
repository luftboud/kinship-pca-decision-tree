import os
import cv2
import numpy as np


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize(face):
    return face.astype("float") / 255.0


def preprocess_file(path, target_size=(100, 100)):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    gray = grayscale(img)

    face = cv2.resize(gray, target_size)
    face = normalize(face)
    flat = face.flatten()
    return flat


def preprocess_dir(directory, target_size=(100, 100)):
    matrices = []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        flat = preprocess_file(path, target_size)
        matrices.append(flat)

    matrices = np.array(matrices)

    return matrices
