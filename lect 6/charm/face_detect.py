import cv2
import numpy as np
from os import path

name = input("Enter your name : ")


cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 50
face_list = []


while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # standard is BGR bt for gray conversion we can use any of them
    faces = classifier.detectMultiScale(gray)

    areas = []

    for face in faces:
        x, y, w, h = face
        area = w * h
        areas.append((area, face))

    areas = sorted(areas, reverse=True)

    if len(areas) > 0:
        face = areas[0][1]
        x, y, w, h = face

        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img,(100,100))
        cv2.imshow("facenew", face_img)
        face_list.append(face_img.flatten())

        count -= 1
        print("loaded", 50 - count)
        if count <= 0:
            break

    # shape = gray.shape
    # small = cv2.resize(gray,(200,200))
    # #cv2.imshow("video", small)
    # big = cv2.resize(small,(shape[1],shape[0]))

    # cv2.imshow("video2",big)
    if cv2.waitKey(1) > 30:
        break
face_list = np.array(face_list)
print(face_list.shape)
name_list = np.full((len(face_list),1),name)
total = np.hstack([name_list,face_list])
if path.exists("faces.npy"):
        data = np.load("faces.npy")
        data = np.vstack([data, total])
else:
        data = total

np.save("faces.npy",data)



cap.release()
cv2.destroyAllWindows()
