import cv2
import os

loc = os.path.dirname(os.path.abspath(__name__))
sample_img_loc=loc + '/computer_vision/facial_identifiction_expression_detection/data/sample.png'
print(sample_img_loc)
gray_img = cv2.cvtColor(cv2.imread(sample_img_loc), cv2.COLOR_RGB2GRAY)

config_loc = loc + '/computer_vision/facial_identifiction_expression_detection/params/haarcascade_frontalface_default.xml'
cascade_clf = cv2.CascadeClassifier( config_loc )
faces = cascade_clf.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    # draw bounding box on frame
    cv2.rectangle(gray_img, (x, y), (x + w, y + h), (100, 255, 0), thickness=2)

cv2.imshow('img',gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()