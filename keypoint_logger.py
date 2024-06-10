import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import traceback
import string
import csv

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

count = 0
labels = list(string.ascii_uppercase)
labels.append('next')
labels.append('space')

label_index = 0
label = labels[label_index]

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

        if hands[0]:
            hand = hands[0]
            x, y, w, h = hand[0]['bbox']
            image = np.array( frame[y - offset:y + h + offset, x - offset:x + w + offset])
            cv2.imshow("hand", image)

        frame = cv2.putText(frame, "label=" + str(label) + "  count=" + str(count), (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break


        if interrupt & 0xFF == ord('n'):
            if((label_index + 1) % 29 == 0):
                label_index = 0
            else:
                label_index += 1 
            label = labels[label_index]
            flag = False
            count = 0

        if interrupt & 0xFF == ord('p'):
            if((label_index - 1) == -1):
                label_index = 28
            else:
                label_index -= 1 
            label = labels[label_index]
            flag = False
            count = 0

        if interrupt & 0xFF == ord('a'):
            if flag:
                flag=False
            else:
                ic=0
                flag=True

        if flag==True:
            if step%3==0:
                resized_frame = cv2.resize(image, (28,28))
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                pixel_values = gray_frame.flatten()
                row = [label_index] + pixel_values.tolist()
                writer.writerow(row)
                count += 1
            step+=1

    except Exception:
        print("==",traceback.format_exc() )

capture.release()
cv2.destroyAllWindows()
file.close()