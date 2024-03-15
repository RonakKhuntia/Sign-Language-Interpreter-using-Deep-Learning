**Sign-Language-Interpreter-using-Deep-Learning**
SDP - IH18 Project

------------------------------------------------------------------------------------------------------------------------------------------------------------

**Contributors**

Ansuman Mohanty
Rounak Kumar Khuntia
Priyadarshini Nayak
Kumar Spandan Pattanayak

------------------------------------------------------------------------------------------------------------------------------------------------------------

**About**

**Sign Language Interpreter using Deep Learning** is a part of our B.Tech final year SDP project. This project was developed to procure a way for individuals with hearing 
impairments to communicate effortlessly in diverse social settings.The core algorithmic approach involves training neural networks on extensive datasets containing diverse sign 
language gestures.TensorFlow is used for efficient training and deployment of models for accurate gesture recognition and OpenCV is integrated to handle image and video processing tasks.

------------------------------------------------------------------------------------------------------------------------------------------------------------

**Dependencies Required** 

1.Python 3.11
2.TensorFlow
3.opencv-python
4.scikit-learn
5.numpy
6.mediapipe

------------------------------------------------------------------------------------------------------------------------------------------------------------

**Instructions** 

1.Install python 3.11 
    Windows --> https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe
    If you have a macOS or Linux you can download your respective installer from https://www.python.org/downloads/release/python-3110/
    **Note** : It is mandatory to have a python version between 3.8 - 3.11 to run all dependencies without any error. Python versions of 3.12 and later 
              don't support mediapipe which is an essential library for this project.

2.Open terminal and run the following code to install all dependencies
    python -m pip install tensorflow opencv-python mediapipe scikit-learn numpy 

------------------------------------------------------------------------------------------------------------------------------------------------------------

**Project Structure**

1. Data-Creation.py collects images for training set.
2. Model-Creation.py creates a model using TensorFlow based on the training set.
3. final.py  uses the trained model to predict sign gestures using live video feed from the camera.

For detailed understanding, follow this youtube tutorial : https://www.youtube.com/watch?v=doDUihpj6ro&list=PLr_UfzFd0m3k8sh4agDt1ZKarDfMEGXcx&t=7905
