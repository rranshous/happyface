# ty: https://github.com/insanecodes/Smile-Detection/blob/main/main.py

# Import Libraries
import cv2
import time
import sys

# Load the cascade
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")

# load the image
image_path = sys.argv.pop(1)
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect for face
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# no faces detected
if len(faces) == 0:
  exit(1)

# detect for smile
smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)

# no smile detected
if len(smiles) == 0:
  exit(1)

# we must have found faces and smiles!
exit(0)
