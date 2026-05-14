#身份识别系统
import cv2 as cv
import os
import numpy as np

IMG_SAVE_PATH = 'face/3/'
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('train.yml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('摄像头连接失败')

name_map = {int(f.split('.')[0]): f.split('.')[1] for f in os.listdir(IMG_SAVE_PATH)}

while True:
    ret, frame = cap.read()
    if not ret:
        print('读帧失败')
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_classifier.detectMultiScale(gray)

    for x, y, w, h in faces:
        img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence > 85:
            name = 'unknown'
        else:
            name = name_map[img_id]
        cv.putText(frame, name, (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 255, 0),
                   thickness=2)
        cv.circle(frame, (x + w // 2, y + h // 2), w // 2, color=(255, 0, 0), thickness=2)

    cv.imshow('face', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()