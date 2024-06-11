import cv2
from cvzone.HandTrackingModule import HandDetector

hd2 = HandDetector(maxHands=1)

def process_hands(image_copy, hand_bbox, placeholder_path="env/cvzone/placeholder.jpg"):
    x, y, w, h = hand_bbox 
    offset = 29 
    image = image_copy[y - offset:y + h + offset, x - offset:x + w + offset]  

    placeholder = cv2.imread(placeholder_path)

    handz = hd2.findHands(image, draw=False, flipType=True)
    if handz[0]:
        hand = handz[0]
        pts = hand[0]['lmList']  

        os = ((400 - w) // 2) - 15
        os1 = ((400 - h) // 2) - 15
        for t in range(0, 4, 1):
            cv2.line(placeholder, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(5, 8, 1):
            cv2.line(placeholder, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(9, 12, 1):
            cv2.line(placeholder, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(13, 16, 1):
            cv2.line(placeholder, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        for t in range(17, 20, 1):
            cv2.line(placeholder, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
        cv2.line(placeholder, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
        cv2.line(placeholder, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
        cv2.line(placeholder, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
        cv2.line(placeholder, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
        cv2.line(placeholder, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

        for i in range(21):
            cv2.circle(placeholder, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

        return pts, placeholder.reshape(1, 400, 400, 3)
