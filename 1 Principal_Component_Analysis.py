# ---- ---- ---- ---- Data Preprocessing (Common Part for Projects 1-4) ---- ---- ---- ----
import mido
import Utilities
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
""" Number of tracks in the melody library, 
    List of titles and singers, 
    Dimension of feature vector for all melodies: """
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

# ---- ---- ---- ---- Datasets for Further Processing (Common Part for Projects 1-4) ---- ---- ---- ----
X0 = np.zeros((N0, D))
for i in range(0, N0):
    X0[i] = Melodies_0[i].degrees_and_pauses.copy()
X1 = np.zeros((N1, D))
for j in range(0, N1):
    X1[j] = Melodies_1[j].degrees_and_pauses.copy()

# ---- ---- ---- ---- Principal Component Analysis ---- ---- ---- ----
from sklearn.decomposition import PCA
# To use the module above you need to install the additional module 'scikit-learn'
import matplotlib.pyplot as plt

pca0 = PCA(n_components=2).fit(X0)
X0_pca = pca0.transform(X0)
# print(X0_pca, '\n')
# print(pca0.components_, '\n')
# print(pca0.explained_variance_ratio_, '\n')
# print(sum(pca0.explained_variance_ratio_))
plt.title("Melody Library C_G_Am_F_x2")
plt.scatter(X0_pca[:, 0],
            X0_pca[:, 1])
for i in range(0, N0):
    plt.text(X0_pca[i, 0], X0_pca[i, 1], Melodies_0[i].title)
# https://towardsdatascience.com/how-to-add-text-labels-to-scatterplot-in-matplotlib-seaborn-ec5df6afed7a
plt.show()

pca1 = PCA(n_components=2).fit(X1)
X1_pca = pca1.transform(X1)
# print(X1_pca, '\n')
# print(pca1.components_, '\n')
# print(pca1.explained_variance_ratio_, '\n')
# print(sum(pca1.explained_variance_ratio_))
plt.title("Melody Library C_G_Am_F_x2")
plt.scatter(X1_pca[:, 0],
            X1_pca[:, 1])
for i in range(0, N1):
    plt.text(X1_pca[i, 0], X1_pca[i, 1], Melodies_1[i].title)
# https://towardsdatascience.com/how-to-add-text-labels-to-scatterplot-in-matplotlib-seaborn-ec5df6afed7a
plt.show()
