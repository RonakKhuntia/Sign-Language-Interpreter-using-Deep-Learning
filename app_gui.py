import cv2
import copy
import traceback
import pyttsx3
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
from model import KeyPointClassifier
import mediapipe as mp
import threading
from app import *

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.speak_engine=pyttsx3.init()
        self.engine_lock = threading.Lock()
        self.speak_engine.setProperty("rate",100)
        voices=self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice",voices[0].id)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.keypoint_classifier_labels = ['Hello','Yes','Dislike','OK','Peace','I Love You','Like','No']
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history = deque(maxlen=16)
        self.use_brect = True

        self.root = tk.Tk()
        self.root.title("Sign Language Interpreter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.state('zoomed')
        self.root.geometry("1300x700")

        self.panel = tk.Label(self.root)
        self.panel.place(x=500, y=3, width=480, height=640)

        self.T = tk.Label(self.root)
        self.T.place(x=450, y=5)
        self.T.config(text="Sign Language Interpreter", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=900, y=585)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=500, y=580)
        self.T1.config(text="Predicted Word :", font=("Courier", 30, "bold"))

        self.speak = tk.Button(self.root)
        self.speak.place(x=700, y=650)
        self.speak.config(text="Speak", font=("Courier", 20), wraplength=100, command=self.speak_fun)

        self.current_symbol = ""

        self.video_loop()

    def video_loop(self):
        try:
            ret, image = self.vs.read()
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True
            self.current_image = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # Hand sign classification
                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                        
                    hand_sign_text = self.keypoint_classifier_labels[hand_sign_id]
                    self.current_symbol = hand_sign_text
            else:
                self.point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, self.point_history)
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

            self.current_image = Image.fromarray(debug_image)

            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            self.panel3.config(text=self.current_symbol, font=("Courier", 30))

        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def speak_fun(self):
        speaking_thread = threading.Thread(target=self.speak_text)
        speaking_thread.start()

    def speak_text(self):
        with self.engine_lock:
            self.speak_engine.say(self.current_symbol)
            self.speak_engine.runAndWait()

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
