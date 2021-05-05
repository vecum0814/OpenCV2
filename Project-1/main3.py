import cv2

img = cv2.imread('01.jpg')
overlay_img = cv2.imread('dices.png', 
    cv2.IMREAD_UNCHANGED) 
    ##png 파일이고 배경이 투명해야 합성이 이쁘게 된다
    ##png 파일을 부를 때는 투명도를 같이 로드하여 합성을 하려고 한다.
    ##그래서 png 파일을 이용하여 오버레이를 할때는 IMREAD_UNCHANGED를 쓰자

overlay_img = cv2.resize(overlay_img, dsize = (150, 150))

##BGR은 색깔은 표시 가능하지만 투명도는 표시하지 못한다.
##하지만 지금 dices.png는 투명도가 있는 이미지기이기 때문에, 
##BGR과 새로 A 채널을 포함하여 총 채널이 4개이다. 
##이 A 채널을 이용하여 각 위치에서 배경 이미지와 오버레이 이미지 중에 어떤 픽셀 값을 따라야 할지 정해준다.

##현재 dices.png에서 주사위가 있는 영역들의 투명도는 다 255이다. -> 그래서 색깔이 보이는것
##하지만 배경 부분은 투명도가 0이기 때문에 다 투명하게 보인다.

overlay_alpha = overlay_img[:, :, 3:] / 255.0
##overlay 할 A 채널 값
##overlay_img에는 [ height, width , channel]의 순서대로 정보가 있다.
## :, -> 모든 정보를 들고온다 -> height && width의 모든 정보를 들고온다 -> 원본 이미지 그대로 들고온다.
## 3: 3번째 채널을 가져온다.
background_alpha = 1.0 - overlay_alpha
##배경의 A 채널 값
## 지금 뭘 한거냐면 그냥 overlay_img랑 background_alpha 각각의 alpha 채널만 건드리고 있는것.
x1 = 100
y1 = 100
x2 = x1 + 150
y2 = y1 + 150

img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] 
+ background_alpha * img[y1:y2, x1:x2]

cv2.imshow('img', img)
cv2.waitKey(0)