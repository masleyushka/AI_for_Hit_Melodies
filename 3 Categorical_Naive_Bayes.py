# ---- ---- ---- ---- Data Preprocessing (Common Part for Projects 1-5) ---- ---- ---- ----
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

# ---- ---- ---- ---- Merging Two Datasets  (Common Part for Projects 3-4) ---- ---- ---- ----
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

# ---- ---- ---- ---- Shuffling the Resulting Dataset ---- ---- ---- ----
idx = np.random.permutation(N)
X = X[idx]
Y = Y[idx]
L_rearranged = []
for k in idx:
    L_rearranged.append(L[k])
L = L_rearranged.copy()

# ---- ---- ---- ---- Splitting Dataset into Train and Test Subsets ---- ---- ---- ----
""" Proportion of the train subset to the whole dataset: """
percentage_train = 0.9
N_train = round(N * percentage_train)
N_test = N - N_train
X_train, X_test = X[:N_train], X[N_train:]
Y_train, Y_test = Y[:N_train], Y[N_train:]
L_train, L_test = L[:N_train], L[N_train:]

# ---- ---- ---- ---- Categorical Naive Bayes ---- ---- ---- ----
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

model = CategoricalNB(min_categories=13)
model.fit(X_train, Y_train)
print("Train score: ", model.score(X_train, Y_train))
print("Test score: ", model.score(X_test, Y_test))
print(model.get_params())
"""https://pyprog.pro/io_functions/set_printoptions.html"""
np.set_printoptions(suppress=True)
print(f'{model.predict_proba(X_test)}')
np.set_printoptions(suppress=False)
P_test = model.predict(X_test)
"""https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html"""
M_test = confusion_matrix(Y_test, P_test)
sns.heatmap(M_test.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('predicted label')
plt.ylabel('true label')
# ConfusionMatrixDisplay.from_predictions(Y_test, P_test)
plt.show()