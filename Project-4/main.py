import cv2
import dlib 

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture('Project-4/videos/01.mp4')
#cap = cv2.VideoCapture('/Users/raylee/Desktop/DeepLearning/Project-4/videos/01.mp4')
sticker_img = cv2.imread('Project-4/imgs/sticker01.png', cv2.IMREAD_UNCHANGED)
# 이렇게 해야 투명도가 포함된 채널도 같이 반영이 된다. 

while True:
    ret, img = cap.read()

    if ret == False:
        break


    dets = detector(img)
    print("number of faces detected:", len(dets)) # 얼굴이 리스트 형태로 저장이 돼 있으니까 리스트의 길이로 사람 명수를 확인

    for det in dets:
        x1 = det.left() - 40
        y1 = det.top() - 50
        x2 = det.right() + 40
        y2 = det.bottom() + 30

        #cv2.rectangle(img, pt1=(x1,y1), pt2=(x2, y2), color=(255, 0, 0), thickness = 2)
        try:
            overlay_img = sticker_img.copy()

            overlay_img = cv2.resize(overlay_img, dsize=(x2 - x1, y2 - y1))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]
        except:
            pass

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break