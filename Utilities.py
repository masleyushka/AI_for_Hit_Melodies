# ---- ---- ---- ---- Importing Dependencies ---- ---- ---- ----
import Charming_Melody
import mido
import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# # PAUSE = float('inf')
# from scipy.cluster.hierarchy import dendrogram, linkage
# import math
# import statistics

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
def list_of_melodies_and_its_parameters(Melody_Library,
                                        number_of_timestamps_in_one_measure,
                                        number_of_measures):
    one_time_unit = 1920 // number_of_timestamps_in_one_measure
    N = len(Melody_Library.tracks) - 1
    L = []
    D = number_of_measures * number_of_timestamps_in_one_measure
    Melodies = []
    for track_number in range(1, N+1):
        degrees_and_pauses = np.zeros(D)
        duration = 0
        for msg in Melody_Library.tracks[track_number]:
            if msg.type == 'note_on':
                if msg.time != 0:
                    duration += msg.time // one_time_unit
            elif msg.type == 'note_off':
                duration_of_degree = msg.time // one_time_unit
                degree = msg.note % 12 + 1
                for i in range(0, duration_of_degree):
                    degrees_and_pauses[duration + i] = degree
                duration += duration_of_degree
        titles_and_singers = Melody_Library.tracks[track_number].name.split(" by ")
        L.append(titles_and_singers[0])
        melody = Charming_Melody.Charming_Melody(
            track_number - 1,
            titles_and_singers[0],
            titles_and_singers[1],
            degrees_and_pauses
        )
        Melodies.append(melody)
    return N, L, D, Melodies