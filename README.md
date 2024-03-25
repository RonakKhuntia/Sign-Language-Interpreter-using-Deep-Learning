<h1>Sing Language Interpreter using Deep Learning I SDP-IH18 Project</h1>

<h2>Contributors</h2>

- [Ansuman Mohanty](https://github.com/Ansuman3152)

- [Rounak Kumar Khuntia](https://github.com/RonakKhuntia)

- [Priyadarshini Nayak](https://github.com/priyu1109)

- [Kumar Spandan Pattanayak](https://github.com/5p7Ro0t)

<h2>About</h2>

<b>" Sing Language Interpreter using Deep Learning "</b> is a part of our B.Tech Final Year SDP Project.The primary objective is to develop an accurate and efficient system capable of translating sign language gestures 
into text, facilitating seamless interaction for the deaf and hard-of-hearing community. The system utilizes neural networks trained on extensive sign language datasets to recognize the 
intricate movements and expressions inherent in sign language. Computer vision techniques are employed to 
capture and interpret gestures accurately, ensuring a high level of precision in the translation process. 

<h2>Requirements</h2>

- mediapipe(requires a python version between 3.8 - 3.11, not supported by other versions)
  
- OpenCV
  
- Tensorflow
  
- scikit-learn
  
- matplotlib

<h2>Instructions</h2>

1.Clone this repo

2.Open terminal and go to the cloned repo directory

3.Your system should have a python version between 3.8 - 3.11 to run all dependencies without any error.<br>
In case you have a different version, uninstall it and install the one from link below.

  Download python 3.11 from below link :

    https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe

2.In terminal, run the following command to install all dependencies

    pip install tensorflow opencv-python mediapipe scikit-learn matplotlib

3.Run the following command to launch open-cv camera window to start detecting hand signs

    python app.py

4.The current model is able to predict 8 hand signs - Hello, Yes, No, Dislike, Like, Ok, Peace, ILoveYou

<img src="https://github.com/RonakKhuntia/Sign-Language-Interpreter-using-Deep-Learning/blob/main/hand-signs.jpg" alt="Alt Text" width="500" height="500">

<h2>Directory Structure</h2>

<h3>app.py</h3>

This is a sample program for inference.
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

<h3>keypoint_classification.ipynb</h3>

This is a model training script for hand sign recognition.

<h3>model/keypoint_classifier</h3>
This directory stores files related to hand sign recognition.
The following files are stored.

- Training data(keypoint.csv)
- Trained model(keypoint_classifier.tflite)
- Label data(keypoint_classifier_label.csv)
- Inference module(keypoint_classifier.py)

<h2>Training</h2>

<b>1.Learning data collection</b>

Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）.
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv"

<b>2.Model training</b>

Open "keypoint_classification.ipynb" in Jupyter Notebook and execute from top to bottom.
To change the number of training data classes, change the value of "NUM_CLASSES" and modify the label of  "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.
