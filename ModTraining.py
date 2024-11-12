import cv2
import numpy as np
import os

# Загрузка каскадного классификатора Хаара для обнаружения лиц
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

data_path = 'Face_Samples_Dataset'
training_data, labels = [], []

for person_id, subdir in enumerate(os.listdir(data_path)):
    person_path = os.path.join(data_path, subdir)
    if not os.path.isdir(person_path):
        continue

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)

        for (x,y,w,h) in faces:
            roi = gray_img[y:y+h,x:x+w]
            roi = cv2.resize(roi, (200, 200))#до этого была ошибка
            training_data.append(roi)
            labels.append(person_id)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.array(training_data), np.array(labels))

model.save('face_recognition_model.yml')







