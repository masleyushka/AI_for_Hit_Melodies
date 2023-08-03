# ---- ---- ---- ---- Importing Dependencies ---- ---- ---- ----
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

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

# ---- ---- ---- ---- Datasets for Processing (Common Part for Projects 1-4) ---- ---- ---- ----
X0 = np.zeros((N0, D))
for i in range(0, N0):
    X0[i] = Melodies_0[i].degrees_and_pauses.copy()
X1 = np.zeros((N1, D))
for j in range(0, N1):
    X1[j] = Melodies_1[j].degrees_and_pauses.copy()

# ---- ---- ---- ---- Hierarchy Clustering ---- ---- ---- ----
""" Here are the control words for the method below:
 -  'single' = Single-Linkage:
    d(clusterA, clusterB) = min distance between any 2 point, 1 from A, 1 from B
    It is not good because it can cause a chaining effect
    About chaining effect
    https://www.youtube.com/watch?v=Kk_5BexOfFY
 -  'complete' = Complete-Linkage:
    d(clusterA, clusterB) = max distance between any 2 point, 1 from A, 1 from B
 -  'ward', 'average', 'centroid':
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Good explanation is here:
    https://www.youtube.com/watch?v=vg1w5ZUF5lA """
method = 'ward'
Z0 = linkage(X0, method)
# print(f"Z0.shape: {Z0.shape}")
plt.title(f"Hierarchy Clustering of Melody Library with '{method}'-method")
dendrogram(Z0, labels=L0)
plt.show()
Z1 = linkage(X1, method)
# print(f"Z1.shape: {Z1.shape}")
plt.title(f"Hierarchy Clustering of Melody Library with '{method}'-Method")
dendrogram(Z1, labels=L1)
plt.show()