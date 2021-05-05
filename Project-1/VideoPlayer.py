import cv2

cap = cv2.VideoCapture('04.mp4')
##cap = cv2.VideoCapture(0) 
##이렇게 하면 내 웹캠을 사용할 수 있음


while True:
    ret, img = cap.read()
    ##img -> 동영상에서 이미지 캡쳐를 한장 한장씩 받아옴
    ##ret -> 동영상이 끝났을 때 이 ret이라는 변수는 False가 d된다.

    if ret == False:
        break

    cv2.rectangle(img, pt1 = (721, 183), pt2 = (878, 465), color = (255, 0, 0), thickness = 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize = (640, 360))
    ##img = img[100:200, 150:250]
    cv2.imshow('result', img)
	


    if cv2.waitKey(1) == ord('q'):
        ##1ms만큼 기다렸다가 다음 프레임 실행, 키보드 입력이 'q'면 break
        break