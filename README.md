<h1>Sign-Language-Interpreter-using-Deep-Learning I SDP - IH18 Project</h1>
<h2>Contributors</h2>

- [Ansuman Mohanty](https://github.com/Ansuman3152)

- [Rounak Kumar Khuntia](https://github.com/RonakKhuntia)

- [Priyadarshini Nayak](https://github.com/priyu1109)

- [Kumar Spandan Pattanayak](https://github.com/5p7Ro0t)

<h2>About</h2>
<b>" Sign Language Interpreter using Deep Learning "</b> is a part of our B.Tech final year SDP project. This project was developed to procure a way for individuals with hearing impairments to communicate effortlessly in diverse social settings.The core algorithmic approach involves training neural networks on extensive datasets containing diverse sign language gestures.TensorFlow is used for efficient training and deployment of models for accurate gesture recognition and OpenCV is integrated to handle image and video processing tasks.

<h2>Requirements</h2>

- mediapipe
  
- OpenCV
  
- Tensorflow
  
- scikit-learn (Only if you want to display the confusion matrix)
  
- matplotlib (Only if you want to display the confusion matrix)

<h2>Instructions</h2>

1.Open terminal and run 'python app.py'.

2.The OpenCV window will open and you can show hand signs.

3.The model will predict the signs and display the sign text on screen.

4.The current model is able to predict 8 hand signs - Hello, Yes, Dislike, OK, Peace, ILoveYou, Like, No


<h2>Directory</h2>

<h3>app.py</h3>
This is a sample program for inference.
In addition, learning data (key points) for hand sign recognition,
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
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

<h3>Hand sign recognition training</h3>

<b>1.Learning data collection</b>

Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）

If you press "0" to "9", the key points will be added to model/keypoint_classifier/keypoint.csv"


<b>2.Model training</b>

Open "keypoint_classification.ipynb" in Jupyter Notebook and execute from top to bottom.

To change the number of training data classes, change the value of "NUM_CLASSES" and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.
