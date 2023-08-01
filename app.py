# ty: https://github.com/insanecodes/Smile-Detection/blob/main/main.py

# Import Libraries
import cv2
import time
import sys

# Load the cascade
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
all_face_cascades = [
  cv2.CascadeClassifier("haarcascade_frontalface_alt.xml"),
  cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml"),
  cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml"),
  cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
]

# load the image
image_path = sys.argv.pop(1)
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect for faces
face_finds = map(lambda fc: fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4), all_face_cascades)
found_faces = 0

# run a few diff classifiers
for face_find in face_finds:
  if len(face_find) != 0:
    found_faces = found_faces + 1

# not enough faces detected
if found_faces < 3:
  exit(1)

# detect for smile
smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)

# no smile detected
if len(smiles) == 0:
  exit(1)

# we must have found faces and smiles!
exit(0)
