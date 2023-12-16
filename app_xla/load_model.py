import cv2
import numpy as np
import time
import mediapipe as mp
import pyttsx3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

class model:
    #Actions that we try to detect
    actions = np.array([ 'none', 'hello', 'i like you', 'nice to meet', '1', '2', '10', 'cat', 'dog', 'sleep', 'look', 'strong'])

        # Thirty videos worth of data
    no_sequences = 30

        # Videos are going to be 30 frames in length
    sequence_length = 30
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30,1662)))
    model.add(Dropout(0.2))
        # model.add(LSTM(64, return_sequences=True, activation='relu'))
        # model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
        # model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    print("1")
    model.load_weights('action.h5')