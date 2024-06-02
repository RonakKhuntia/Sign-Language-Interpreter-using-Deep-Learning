import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback
import string
import csv

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

count = len(oss.listdir("data/images/A/"))
c_dir = list(string.ascii_uppercase)
c_dir.append('next')
c_dir.append('backspace')
c_dir.append('space')

dir_index = 0
curr_dir = c_dir[dir_index]

offset = 15
step = 1
flag=False
ic=0

placeholder=np.ones((400,400),np.uint8)*255
cv2.imwrite("placeholder.jpg",placeholder)

train_dataset = 'data/dataset/sign_mnist_train.csv'
file = open(train_dataset, mode='a', newline='')
writer = csv.writer(file)

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands= hd.findHands(frame, draw=False, flipType=True)
        placeholder = cv2.imread("placeholder.jpg")

        if hands[0]:
            hand = hands[0]
            x, y, w, h = hand[0]['bbox']
            image = np.array( frame[y - offset:y + h + offset, x - offset:x + w + offset])
            handz = hd2.findHands(image, draw=True, flipType=True)
            if handz[0]:
                hand = handz[0]
                pts = hand[0]['lmList']
                os=((400-w)//2)-15
                os1=((400-h)//2)-15
                for t in range(0,4,1):
                    cv2.line(placeholder,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(5,8,1):
                    cv2.line(placeholder,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(9,12,1):
                    cv2.line(placeholder,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(13,16,1):
                    cv2.line(placeholder,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                for t in range(17,20,1):
                    cv2.line(placeholder,(pts[t][0]+os,pts[t][1]+os1),(pts[t+1][0]+os,pts[t+1][1]+os1),(0,255,0),3)
                cv2.line(placeholder, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (0, 255, 0), 3)
                cv2.line(placeholder, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (0, 255, 0), 3)
                cv2.line(placeholder, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)
                cv2.line(placeholder, (pts[0][0]+os, pts[0][1]+os1), (pts[5][0]+os, pts[5][1]+os1), (0, 255, 0), 3)
                cv2.line(placeholder, (pts[0][0]+os, pts[0][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)

                for i in range(21):
                    cv2.circle(placeholder,(pts[i][0]+os,pts[i][1]+os1),2,(0 , 0 , 255),1)

                skeleton0=np.array(placeholder)

        skeleton1=np.array(placeholder)

        cv2.imshow("skeleton1",skeleton1)

        frame = cv2.putText(frame, "dir=" + curr_dir + "  count=" + str(count), (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break


        if interrupt & 0xFF == ord('n'):
            if((dir_index + 1) % 29 == 0):
                dir_index = 0
            else:
                dir_index += 1 
            curr_dir = c_dir[dir_index]
            flag = False
            count = len(oss.listdir("data/images/" + (curr_dir) + "/"))

        if interrupt & 0xFF == ord('p'):
            if((dir_index - 1) == -1):
                dir_index = 28
            else:
                dir_index -= 1 
            curr_dir = c_dir[dir_index]
            flag = False
            count = len(oss.listdir("data/images/" + (curr_dir) + "/"))

        if interrupt & 0xFF == ord('a'):
            if flag:
                flag=False
            else:
                ic=0
                flag=True

        if flag==True:
            if ic==180:
                flag=False
            if step%3==0:
                cv2.imwrite("data/images/" + (curr_dir) + "/" + str(count) + ".jpg",
                            skeleton1)
                resized_frame = cv2.resize(image, (28, 28))
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                pixel_values = gray_frame.flatten()
                label = dir_index
                row = [label] + pixel_values.tolist()
                writer.writerow(row)
                count += 1
                ic += 1
            step+=1

    except Exception:
        print("==",traceback.format_exc() )

capture.release()
cv2.destroyAllWindows()
file.close()