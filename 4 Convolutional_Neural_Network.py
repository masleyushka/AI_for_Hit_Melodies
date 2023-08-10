# ---- ---- ---- ---- Data Preprocessing (Common Part for Projects 1-4) ---- ---- ---- ----
import Utilities
import mido
import numpy as np

Melody_Library_0 = mido.MidiFile('C_G_Am_F_x2.mid')
Melody_Library_1 = mido.MidiFile('Am_F_C_G_x2.mid')
""" A resolution of MIDI track can be set as high as 1,920 PPQ (pulses per quarter note).
    See details here 
    https://www.dummies.com/article/technology/software/music-recording-software/general-music-recording-software/midi-time-code-and-midi-clock-explained-179973/
    In what follows, we will assume that 
    - each song contains 10 measures, 
    - each measure contains 16 timestamps.
    So, we need to divide 1920 by 16. 
    We can choose another number of timestamps, but it must be a divisor of 1920.
    Here they are
    1, 2, 3, 4, 5, 6, 8,
    10, 12, 15, {16}, 20, 24, 30, 32, 40, 48, 60, 64, 80, 96,
    120, 128, 160, 192, 240, 320, 384, 480, 640, 960,
    1920. """
number_of_timestamps_in_one_measure = 16
number_of_measures = 10
""" The following utility returns
 -  Number of tracks in a given melody library (equals 100 for both "C_G_Am_F" and Am_F_C_G), 
 -  List of titles and singers, 
 -  Dimension of feature vectors (equals 160),
 -  List of Charming_Melody-instances. """
N0, L0, D, Melodies_0 = Utilities.list_of_melodies_and_its_parameters(
    Melody_Library_0,
    number_of_timestamps_in_one_measure,
    number_of_measures)
# for item in Melodies_0:
#     print(item.degrees_and_pauses)
N1, L1, D, Melodies_1 = Utilities.list_of_melodies_and_its_parameters(
    Melody_Library_1,
    number_of_timestamps_in_one_measure,
    number_of_measures)
# for item in Melodies_1:
#     print(item.degrees_and_pauses)

# ---- ---- ---- ---- Merging Two Datasets (Common Part for Projects 3-4) ---- ---- ---- ----
N = N0 + N1
L = L0 + L1
X = np.zeros((N, D))
for i in range(0, N0):
    X[i] = Melodies_0[i].degrees_and_pauses.copy()
for j in range(0, N1):
    X[N0+j] = Melodies_1[j].degrees_and_pauses.copy()
Y0 = np.zeros(N0)
Y1 = np.ones(N1)
Y = np.hstack([Y0, Y1])

# ---- ---- ---- ---- Shuffling the Resulting Dataset (Common Part for Projects 3-4) ---- ---- ---- ----
idx = np.random.permutation(N)
X = X[idx]
Y = Y[idx]
L_rearranged = []
for k in idx:
    L_rearranged.append(L[k])
L = L_rearranged.copy()

# ---- ---- ---- ---- Splitting Dataset into Train and Test Subsets (Common Part for Projects 3-4) ---- ---- ---- ----
""" Proportion of the train subset to the whole dataset: """
percentage_train = 0.9
N_train = round(N * percentage_train)
N_test = N - N_train
X_train, X_test = X[:N_train], X[N_train:]
Y_train, Y_test = Y[:N_train], Y[N_train:]
L_train, L_test = L[:N_train], L[N_train:]

# ---- ---- ---- ---- Convolutional Neural Network ---- ---- ---- ----
""" The following code is based on the courses
 1) "TensorFlow 2 Beginner Course"
    by Murat Karakaya from Youtube
    https://www.youtube.com/watch?v=eMMZpas-zX0&list=PLqnslRFeH2Uqfv1Vz3DqeQfy0w20ldbaV&index=5
    https://github.com/patrickloeber/tensorflow-course/blob/master/05_cnn.py
 2) "TensorFlow 2 Deep Learning"
    by Minsuk Heo from Youtube
    https://www.youtube.com/playlist?list=PLVNY1HnUlO25XeZstpj7m-2RTtyOhv6hO
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from keras import layers
import datetime

# Normalization: 0, ..., 12 --> 0, ..., 1
X_train = X_train / 12
X_test = X_test / 12

FILTERS = 256
KERNEL_SIZE = 5
ACTIVATION = 'tanh'
DROPOUT_RATE = 0.3

model = keras.models.Sequential(name="C_G_Am_F_vs_Am_F_C_G_as_Conv1D")
model.add(layers.Input(shape=(D, 1)))  # the shape is (160, 1)

model.add(layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION))  # (156, 1)
model.add(layers.MaxPool1D(2))  # the shape is (156/2 = 78, 1)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(DROPOUT_RATE))

model.add(layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION))  # (74, 1)
model.add(layers.MaxPool1D(2))  # the shape is (74/2 = 37, 1)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(DROPOUT_RATE))

model.add(layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION))  # (33, 1)
model.add(layers.MaxPool1D(2))  # the shape is (33/2 = 16, 1)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(DROPOUT_RATE))

model.add(layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION))  # (12, 1)
model.add(layers.MaxPool1D(2))  # the shape is (12/2 = 6, 1)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(DROPOUT_RATE))

model.add(layers.Flatten())
model.add(layers.Dense(2))
model.add(layers.BatchNormalization())
print(model.summary())

# ---- ---- ---- ---- Loss Function and Optimizer ---- ---- ---- ----
LOSS = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
OPTIM = keras.optimizers.SGD(ema_momentum=0.8, learning_rate=0.1)
METRICS = ["accuracy"]
model.compile(optimizer=OPTIM, loss=LOSS, metrics=METRICS)

# ---- ---- ---- ---- Training ---- ---- ---- ----
BATCH_SIZE = 20
EPOCHS = 30
X_train = X_train.reshape(N_train, D, 1)
Y_train = Y_train.reshape(N_train, 1)
model.fit(X_train,
          Y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=2)

now = datetime.datetime.now()
filepath = 'Results/'+now.strftime("%Y-%m-%d %H-%M-%S ")+'CNN'
keras.models.save_model(model=model, filepath=filepath)

# ---- ---- ---- ---- Evaluation  ---- ---- ---- ----
X_test = X_test.reshape(N_test, D, 1)
Y_test = Y_test.reshape(N_test, 1)
model.evaluate(X_test,
               Y_test,
               batch_size=BATCH_SIZE,
               verbose=2)
# the_same_model = keras.models.load_model(filepath=filepath)
# the_same_model.evaluate(X_test,
#                         Y_test,
#                         batch_size=batch_size,
#                         verbose=2)
