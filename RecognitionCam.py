import cv2
import numpy as np
import mss
import mss.tools

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if faces == ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = gray[y:y + h, x:x + w]
        return img, roi

    return img, []


#Импорт нашей модели
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_recognition_model.yml')

#Каскадный классификатор Хаара
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Присваиваем каждому ID своё имя
names = {0: "Jhan654", 1: "Person 2"}

#Выбор камеры захвата 0(первая по умолчанию)
cap = cv2.VideoCapture(0)

#Цикл захвата видео с камеры
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image, face = face_detector(frame)

    try:
        face = cv2.resize(face, (500, 500))
        label, confidence = model.predict(face)
        print(f'Confidence: {confidence}, Label: {label}')

        if confidence < 50:
            name = names[label]
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'{confidence:.2f}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(image, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except:
        cv2.putText(image, "No Face Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', image)


    # Нажмите Enter для выхода
    if cv2.waitKey(1) == 13:
        break

    #Закрытие камеры
    cap.release()
    cv2.destroyAllWindows()









