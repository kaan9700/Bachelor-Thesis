import numpy as np
import matplotlib.pyplot as plt
from functions import load_np_array_pickle, augment_data_list
import time
from scipy.signal import find_peaks

def show_plots(epochs):
    start = 0
    end = 9
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(10, 5))
    raw_data = load_np_array_pickle('../files/windows_files/windows_flatlines.pickle')
    values = list(range(int(start), int(end) + 1))

    data_minmax = [el['data'] for el in raw_data]
    data_minmax = np.array([np.min(data_minmax), np.max(data_minmax)])
    ymin = np.min(data_minmax)
    ymax = np.max(data_minmax)

    while values:
        values = list(range(int(start), int(end) + 1))
        k = 0

        # Elemente entfernen, die größer als x sind
        values = [val for val in values if val <= len(epochs)-1]
        print(values)
        for i in range(2):
            for j in range(5):
                if k < len(values) and k < len(epochs):
                    ax[i, j].clear()
                    data = epochs[values[k]]['data']
                    std = np.std(data) # Standardabweichung berechnen
                    ax[i, j].plot(data)
                    ax[i, j].set_ylim(ymin, ymax)
                    ax[i, j].set_title(f"Std: {std:.3f}") # Titel mit Standardabweichung
                else:
                    ax[i, j].set_visible(False)
                k += 1

        #damit die Schleife unterbrochen wird
        if len(values) != 10:
            values = []

        start += 10
        end += 10

        plt.draw()
        plt.pause(0.1)
        time.sleep(2)


if __name__ == '__main__':
    predicted = load_np_array_pickle('../files/windows_files/windows_flatlines.pickle')
    """
        t = load_np_array_pickle('../files/windows_files/final_windows.pickle')
    aug = augment_data_list(t, 50)
    """

    test_fl = load_np_array_pickle('../files/windows_files/test_del.pickle')
    show_plots(test_fl)
