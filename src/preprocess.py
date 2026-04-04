import os
import cv2

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def normalize(face):
    return face.astype("float") / 255.0

def preprocess(directory, target_size=(100, 100)):
    matrices = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        gray = grayscale(img)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        if len(faces) == 0:
           continue

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, target_size)

        face = normalize(face)
        matrices.append(face)
    return matrices