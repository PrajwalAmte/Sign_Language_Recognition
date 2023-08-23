# American Sign Language Recognition Project

![ASL Gesture](sign_mnist/asl_sign.png)

## Project Overview

This project focuses on two main parts: training a model to recognize American Sign Language (ASL) gestures using a dataset, and predicting gestures through a camera feed while converting predictions into speech.

## Part 1: Model Training (`model.py`)

The `model.py` script is responsible for training a convolutional neural network (CNN) model to recognize ASL gestures. The key steps include:

1. **Importing necessary libraries.**
2. **Loading and preprocessing the ASL gesture dataset.**
3. **Defining and compiling the CNN model architecture.**
4. **Training the model using the preprocessed dataset.**
5. **Saving the trained model as `smnist.h5`.**

To train the model, execute:

```
python model.py
```

## Part 2: Gesture Prediction and Speech Conversion (`prediction.py`)

The `prediction.py` script predicts ASL gestures in real-time through a camera feed. It further converts these predictions into speech using the Google Text-to-Speech (gTTS) library. The script's steps are as follows:

1. **Importing Required Libraries**: Import the necessary libraries for the script.

2. **Loading the Pre-trained CNN Model**: Load the pre-trained CNN model (`smnist.h5`) for gesture recognition.

3. **Initializing the Camera Feed**: Initialize the camera feed to capture video frames.

4. **Capturing and Processing Video Frames**: Continuously capture and process video frames from the camera.

5. **Gesture Prediction**: Press the **Space** key to predict the ASL gesture. The predicted gesture will be displayed on the screen.

6. **Closing the Window**: Press the **ESC** key to close the video feed window.

7. **Converting Predictions into Speech using gTTS**: Utilize the gTTS library to convert predictions into speech.

8. **Playing the Generated Speech**: Play the generated speech using an appropriate audio player.


To predict and convert ASL gestures, run the following command:


```
python prediction.py
```

## Requirements

- Python (version 3.x)
- Required libraries (install using pip):

```bash
pip install numpy pandas matplotlib seaborn keras scikit-learn opencv-python mediapipe gTTS
```

## Acknowledgements

This project makes use of the Mediapipe library for hand gesture recognition and the Google Text-to-Speech (gTTS) library for speech synthesis. The ASL gesture dataset used for this project was obtained from [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) and serves as the primary dataset for training and testing the ASL gesture recognition model.

## License

This project is licensed under the MIT License. Refer to the `LICENSE` file for more information.

Feel free to contribute and adapt this project to enhance accessibility and communication for individuals using American Sign Language.


## License

This project is licensed under the MIT License. Refer to the `LICENSE` file for more information.

Feel free to contribute and adapt this project to enhance accessibility and communication for individuals using American Sign Language.
