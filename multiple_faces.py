import cv2
import matplotlib.pyplot as plt

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# OpenCV deals with BGR but Matplotlib deals with RGB
image = cv2.imread('./data/people.jpg')

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

for (x, y, width, height) in detected_faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)

# OpenCV uses BGR and matplotlib uses RGB
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()