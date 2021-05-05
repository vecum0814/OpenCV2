import cv2

img = cv2.imread('01.jpg')




cv2.imshow('img',img) ## 이거만 띄우면 띄웠다가 사라짐

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X: " + str(x) + ", Y: " + str(y))
    else:
        pass
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('img', onMouse)

cv2.waitKey(0) ##이렇게 해주면 무한정으로 키 입력이 있을때까지 기다림