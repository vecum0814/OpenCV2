from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

facenet = cv2.dnn.readNet('models/deploy.prototxt', 
'models/res10_300x300_ssd_iter_140000.caffemodel')
##얼굴 영역 탐지 모델 load
model = load_model('models/mask_detector.model')
##mask detecting model 그리고 load_models = 텐서플로에서 쓰는 함수형

##cap = cv2.VideoCapture('videos/calmDownMan.mov')
cap = cv2.VideoCapture('videos/04.mp4')
ret, img = cap.read()


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))


while True:
    ret, img = cap.read()
    ##img -> 동영상에서 이미지 캡쳐를 한장 한장씩 받아옴
    ##ret -> 동영상이 끝났을 때 이 ret이라는 변수는 False가 d된다.

    if ret == False:
        break

    h, w, c = img.shape
    # 이미지 전처리하기
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104., 177., 123.))

    # 얼굴 영역 탐지 모델로 추론하기
    facenet.setInput(blob)
    dets = facenet.forward()

    # 각 얼굴에 대해서 반복문 돌기
    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]

        if confidence < 0.5:
          continue

        # 사각형 꼭지점 찾기
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]

        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        ##predict -> opencv의 forward와 똑같다. 즉, 넘겨주는것
        ## mask + nomask = 1, 즉 mask를 0.7의 확률로 썼다고 예측이 되면, nomask에는 0.3이 들어간다.
        mask, nomask = model.predict(face_input).squeeze()

        if mask > nomask:
            color = (0, 255, 0)
        else:
            color = (0,0,255)
        # 사각형 그리기
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color)

    
    cv2.imshow('result', img)
    out.write(img)


	


    if cv2.waitKey(1) == ord('q'):
        ##1ms만큼 기다렸다가 다음 프레임 실행, 키보드 입력이 'q'면 break
        break

cap.release()
out.release()
cv2.destroyAllWindows()


