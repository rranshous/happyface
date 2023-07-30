# ty: https://github.com/insanecodes/Smile-Detection/blob/main/main.py

# Import Libraries
import cv2
import time
import sys

# Load the cascade
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# load the image
image_path = sys.argv.pop(1)
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect for smile
smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)

# output success / failure
if smile.any():
    sys.stdout.write("smile found")
    exit(0)
else:
    sys.stdout.write("smile not found")
    exit(1)
