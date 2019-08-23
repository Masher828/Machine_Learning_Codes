import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

data = np.load("My_faces.npy")

print(data.shape)

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier(4)
model.fit(X, y)

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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
        face_img = cv2.resize(face_img, (200, 200))

        # print(res)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 3)
        flat = face_img.flatten()

        res = model.predict([flat])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for names in res:
            cv2.putText(gray, str(names), (x, y), font, 1, (255, 255, 255), 2)

        print(res)
        cv2.imshow("face", gray)

    # cv2.imshow("video2",big)
    if cv2.waitKey(1) > 30:
        break

cap.release()
cv2.destroyAllWindows()
