import cv2
import matplotlib.pyplot as plt

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
image = cv2.imread('./data/girl_face.jpeg')
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detect_faces = cascade_classifier.detectMultiScale(grey_image, scaleFactor=1.1, minNeighbors=10, minSize=(10,10))
#plt.imshow(grey_image, cmap='gray')
#plt.show()

for (x, y, width, height) in detect_faces:
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()