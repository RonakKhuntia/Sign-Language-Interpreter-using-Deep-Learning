<h1>Sign-Language-Interpreter-using-Deep-Learning I SDP - IH18 Project</h1>

<h2><b>Contributors</b></h2>

Ansuman Mohanty

Rounak Kumar Khuntia

Priyadarshini Nayak

Kumar Spandan Pattanayak

<h2><b>About</b></h2>

**Sign Language Interpreter using Deep Learning** is a part of our B.Tech final year SDP project. This project was developed to procure a way for individuals with hearing 
impairments to communicate effortlessly in diverse social settings.The core algorithmic approach involves training neural networks on extensive datasets containing diverse sign 
language gestures.TensorFlow is used for efficient training and deployment of models for accurate gesture recognition and OpenCV is integrated to handle image and video processing tasks.

<h2><b>Dependencies</b></h2>

1.Python 3.11

2.TensorFlow

3.opencv-python

4.scikit-learn

5.numpy

6.mediapipe

<h2><b>Instructions</b></h2>

1.Install python 3.11 

Windows  
    
    https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe
    
If you have a macOS or Linux you can download your respective installer from this link
    
    https://www.python.org/downloads/release/python-3110/
    
**Note** : It is mandatory to have a python version between 3.8 - 3.11 to run all dependencies without any error. Python versions of 3.12 and later 
              don't support mediapipe which is an essential library for this project.

2.Open terminal and run the following code to install all dependencies

Windows

    python -m pip install tensorflow opencv-python mediapipe scikit-learn numpy 

macOS and Linux

    pip install tensorflow opencv-python mediapipe scikit-learn numpy

<h2><b>Project Structure</b></h2>

1. Data-Creation.py collects images for training set.
 
2. Model-Creation.py creates a model using TensorFlow based on the training set.
 
3. final.py uses the trained model to predict sign gestures using live video feed from the camera.

4. action.h5 is a prebuilt model that is able to predict three actions - hello, thankyou, iloveyou

For detailed understanding, follow this youtube tutorial : https://www.youtube.com/watch?v=doDUihpj6ro&list=PLr_UfzFd0m3k8sh4agDt1ZKarDfMEGXcx&t=7905
