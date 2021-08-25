import cap
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'C:\\Users\\milan\\Desktop\\face detection\\face recognizatuion project\\images'

images = []
personName = []

myList = os.listdir(path)
print(myList)

for current_images in myList:
    cu_image = cv2.imread(f'{path}/{current_images}')
    images.append(cu_image)
    personName.append(os.path.splitext(current_images)[0])
print(personName)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = faceEncodings(images)
print("all encoding complete...")

def attandance(name):
    with open('C:\\Users\\milan\Desktop\\face detection\\face recognizatuion project\\attandance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []

        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}')


# cameara reading
cap = cv2.VideoCapture(0)
# id is 0 for laptop nelse 1


while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2),(255, 255, 255), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2)
            attandance(name)

    cv2.imshow("camera", frame)
    if cv2.waitKey(100) == 13:
        break

cap.release()
cv2.destroyAllWindows()
