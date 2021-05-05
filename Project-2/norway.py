import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')
net2 = cv2.dnn.readNetFromTorch('models/instance_norm/the_scream.t7')
cap = cv2.VideoCapture('imgs/Car.mp4')



while True:
    ret, img = cap.read()
    ##img -> 동영상에서 이미지 캡쳐를 한장 한장씩 받아옴
    ##ret -> 동영상이 끝났을 때 이 ret이라는 변수는 False가 된다.

    if ret == False:
        break

    h, w, c = img.shape

    img = cv2.resize(img, dsize=(500, int(h / w * 500)))

    MEAN_VALUE = [103.939, 116.779, 123.680]
    blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

    net.setInput(blob)
    output = net.forward()

    output = output.squeeze().transpose((1, 2, 0))

    output += MEAN_VALUE
    output = np.clip(output, 0, 255)
    output = output.astype('uint8')

    net2.setInput(blob)
    output2 = net2.forward()

    output2 = output2.squeeze().transpose((1, 2, 0))

    output2 += MEAN_VALUE
    output2 = np.clip(output2, 0, 255)
    output2 = output2.astype('uint8')

    output = output[:, :250]
    output2 = output2[:, 250:500]
    output3 = np.concatenate([output, output2], axis = 1)


    ##cv2.rectangle(img, pt1 = (721, 183), pt2 = (878, 465), color = (255, 0, 0), thickness = 2)
    ##img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##img = cv2.resize(img, dsize = (640, 360))
    ##img = img[100:200, 150:250]
    cv2.imshow('result', output3)
    cv2.imshow('original', img)
	


    if cv2.waitKey(1) == ord('q'):
        ##1ms만큼 기다렸다가 다음 프레임 실행, 키보드 입력이 'q'면 break
        break