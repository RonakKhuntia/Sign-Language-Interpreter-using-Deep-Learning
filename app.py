import numpy as np
import cv2
import os
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from env.cvzone.HandProcessing import process_hands
from env.keras.model.labels import process_label
import tkinter as tk
import threading
from PIL import Image, ImageTk

hd = HandDetector(maxHands=1)

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('data/model/sign_language_model.h5')
        self.speak_engine=pyttsx3.init()
        self.engine_lock = threading.Lock()
        self.speak_engine.setProperty("rate",130)
        voices=self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice",voices[0].id)
        
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[]
        for i in range(10):
             self.ten_prev_char.append(" ")

        self.root = tk.Tk()
        self.root.title("Sign Language Interpreter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.state('zoomed')
        self.root.geometry("1300x700")

        self.panel = tk.Label(self.root) #OpenCV display panel
        self.panel.place(x=100, y=3, width=480, height=640)

        self.hand_signs_image = "hand_signs.jpg"
        self.load_image()
        self.panel2 = tk.Label(self.root, image=self.img)
        self.panel2.place(x=700, y=70, width=500, height=500)
        
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language Interpreter", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=280, y=585)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=580)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=260, y=632)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=632)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.panel6 = tk.Label(self.root)  # Accuracy
        self.panel6.place(x=260, y=682)

        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=682)
        self.T4.config(text="Accuracy :", font=("Courier", 30, "bold"))

        self.speak = tk.Button(self.root)
        self.speak.place(x=1400, y=600)
        self.speak.config(text="Speak", font=("Courier", 20), wraplength=100, command=self.speak_fun)

        self.clear = tk.Button(self.root)
        self.clear.place(x=1400, y=670)
        self.clear.config(text="Clear", font=("Courier", 20), wraplength=100, command=self.clear_fun)

        self.delete = tk.Button(self.root)
        self.delete.place(x=1400, y=740)
        self.delete.config(text="Delete", font=("Courier", 20), wraplength=100, command=self.backspace)

        self.str = " " #Sentence variable
        self.current_symbol = ""  #Current Symbol variable
        self.acc = 0 #Accuracy Variable

        self.video_loop()

    def load_image(self):
        img = Image.open(self.hand_signs_image)
        img = img.resize((400, 505), Image.Resampling.LANCZOS)
        self.img = ImageTk.PhotoImage(img)


    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy=np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands[0]:
                hand = hands[0]
                self.pts, roi = process_hands(cv2image_copy,hand[0]['bbox'])
                prediction = np.array(self.model.predict(roi), dtype='float32')
                self.acc = "{:.2f}".format(np.max(prediction)*100)
                pred_char = process_label(prediction, self.pts, self.model)
                self.update_str(pred_char)

                self.panel3.config(text=self.current_symbol, font=("Courier", 30)) #Current Symbol

            self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025) #Sentence

            self.panel6.config(text=self.acc, font=("Courier", 30), wraplength=1025) #Accuracy
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def speak_fun(self):
        speaking_thread = threading.Thread(target=self.speak_text)
        speaking_thread.start()

    def speak_text(self):
        with self.engine_lock:
            self.speak_engine.say(self.str)
            self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str=" "

    def backspace(self):
        self.str=self.str[0:-1]

    def update_str(self, ch):
        if ch=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]

        self.prev_char=ch
        self.current_symbol=ch
        self.count += 1
        self.ten_prev_char[self.count%10]=ch

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

print("Starting Application...")

(Application()).root.mainloop()