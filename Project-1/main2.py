import cv2

img = cv2.imread('01.jpg')

print(img)
print(img.shape)



cv2.rectangle(img, pt1 = (259, 89), pt2 = (380, 348),
    color = (255, 0, 0), thickness = 2
    )  ##pt1: 사각형의 왼쪽 위 좌표 pt2: 사각형의 오른쪽 아래 좌표

cv2.circle(img, center = (320, 220), radius = 100, 
    color = (0, 0, 255), thickness = 3)

cropped_img = img[89:348, 259:380] 
##이미지를 자를 때는 xy 순서가 아니라 yx 순서대로 쓴다

img_resized = cv2.resize(img, (512, 256))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('result', img_rgb)
##cv2.imwrite('res.jpg', img_rgb) ## 저장 가능
cv2.waitKey(0)


cv2.imshow('resized', img_resized)
cv2.imshow('crop', cropped_img)
cv2.imshow('img',img) ## 이거만 띄우면 띄웠다가 사라짐
cv2.waitKey(0) ##이렇게 해주면 무한정으로 키 입력이 있을때까지 기다림