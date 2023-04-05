import mne
import os
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import numpy as np
import pickle
import time
from scipy.signal import find_peaks

def is_linear(data, threshold):
    std = np.std(data)
    if std > threshold:
        return True
    else:
        return False


def clear_flatlines(data, threshold1):
    windows = data

    # Entferne die Elemente, die nicht linear sind, in einer separaten Schleife
    to_delete1 = set()
    for idx1, window in enumerate(windows):
        if not is_linear(window['data'], threshold1):
            to_delete1.add(idx1)
    windows = np.delete(windows, list(to_delete1), axis=0)
    deleted_windows = [data[i] for i in to_delete1]

    return windows, deleted_windows


def show_plots(epochs):
    start = 0
    end = 9
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(10, 5))
    raw_data = load_np_array_pickle('../files/epochs_files/rawdata.pickle')

    values = list(range(int(start), int(end) + 1))

    data_minmax = [el['data'] for el in epochs]

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


def find_cut_peaks(windows, len, marg):
    """
    Diese Funktion überprüft, ob in den Fenstern ein 'abgeschnittener' Peak vorhanden ist.
    Es wird überprüft, ob der Peak zu nah am Rand des Fensters ist.
    Die Funktion gibt zwei Listen zurück: eine Liste mit Fenstern ohne abgeschnittene Peaks und
    eine Liste mit nur den abgeschnittenen Peaks Fenstern.
    """
    window_len = len
    margin = marg  # Anzahl der Punkte, die der Peak vom Fensterrand entfernt sein muss, um nicht abgeschnitten zu sein
    good_windows = []
    cut_windows = []
    for window in windows:
        peak_idx = np.argmax(window['data'])
        if peak_idx <= margin or peak_idx >= window_len - margin:
            cut_windows.append(window)
        else:
            good_windows.append(window)
    return good_windows, cut_windows


def create_window_dict(all_windows, raw):
    window_dict = []
    for idx, channel in enumerate(all_windows):
        for window in channel:
            window_dict.append({'channel': raw.ch_names[idx], 'nr': id, 'time': window['time'], 'data': window['data']})
    return window_dict


def remove_neighbors(data):
    to_delete = []  # Hier werden die zu löschenden Objekte gespeichert
    epochs = data.copy()
    i = 0
    while i < len(data):
        current = data[i]
        consecutive = [current]
        for j in range(i + 1, len(data)):
            if data[j]['nr'] == current['nr'] + 1:
                current = data[j]
                consecutive.append(current)
            else:
                break
        if len(consecutive) > 1:
            # Erstelle Subplot mit der Anzahl der aufeinanderfolgenden Elemente
            raw_data = load_np_array_pickle('./epochs_files/rawdata.pickle')

            data_minmax = raw_data[:-1]

            data_minmax = np.array([np.min(data_minmax), np.max(data_minmax)])
            ymin = np.min(data_minmax)
            ymax = np.max(data_minmax)

            fig, axs = plt.subplots(1, len(consecutive), figsize=(5 * len(consecutive), 5))

            for k in range(len(consecutive)):
                axs[k].plot(consecutive[k]['data'])
                axs[k].set_title(f"nr: {consecutive[k]['nr']}")
                axs[k].set_ylim(ymin, ymax)

            plt.show()
            print(len(consecutive))
            # Auswahl und Entfernung der Fenster
            print('Geben Sie die Nr ein die entfernt werden sollen\nBei mehreren nummern, trennen Sie die Zahlen mit einem Komma')
            numbers = input()

            # Erstelle ein Array aus den Zahlen
            try:
                numbers = numbers.split(',')
            except AttributeError:
                pass

            # Entferne Leerzeichen zwischen den Zahlen
            numbers = [int(elem.strip()) for elem in numbers]

            for obj in consecutive:
                if obj['nr'] in numbers:
                    to_delete.append(obj)
        i += len(consecutive)

    for obj in to_delete:
        epochs.remove(obj)

    return epochs


def get_information(file):

    if file.endswith('.eeg'):
        prefix, file_ext = os.path.splitext(file)
        eeg_file = '../../files/eeg_files/' + prefix+'.vhdr'
        raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
        scan_durn = raw._data.shape[1] / raw.info['sfreq']
        return raw.info['sfreq'], raw.info['ch_names'], scan_durn, raw


def show_all_plots(raw):
    plt.close('all')
    raw.plot()


def create_windows(overlap_val, window_size, raw_data, sample_rate):
    window_size = int(window_size)
    average_overlap = float(overlap_val)
    step_size = int(window_size * (1 - average_overlap))  # Berechne die Schrittgröße
    windows_set = []

    for data in raw_data:
        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            start_time = i / sample_rate
            end_time = (i + window_size) / sample_rate
            time_range = (start_time, end_time)
            windows.append({'time': time_range, 'data': data[i:i + window_size]})
        windows_set.append(windows)

    return windows_set



def eeg_filter(ica, notch, low, high, filename, norm, smoothing):
    name, ext = os.path.splitext(filename)
    new_filename = '../../files/eeg_files/' + name + '.vhdr'
    raw = mne.io.read_raw_brainvision(new_filename, preload=True)

    if ica == 1:
        print("ICA")
        # Data preprocessing
        try:
            # Create ICA object
            ica = ICA(n_components=10, random_state=42)
            # Fit ICA on raw data
            ica.fit(raw)
            # Identify and remove undesirable components
            ica.exclude = [ica.labels_['eye_blink', 'muscle_activity', 'heartbeat']]
            # Apply ICA to raw data
            ica.apply(raw)
        except KeyError:
            pass

    if notch == 1:
        notch_frequencies = [50, 60]
        # Anwendung des Notch-Filters
        raw = raw.notch_filter(notch_frequencies)

    if low != 0 and high != 0:
        # Definieren Sie die Filtergrenzen (z.B. 0,5 - 30 Hz)
        l_freq = float(low)
        h_freq = float(high)
        # Anwenden des Butterworth-Filters
        raw.filter(l_freq, h_freq, fir_design='firwin')

    smoothed_data = []
    if norm == 1:
        # Extrahieren Sie die Rohdaten für jeden Kanal
        data_minmax = raw.get_data()[:-1]
        data_minmax = np.array([data_minmax.min(), data_minmax.max()])
        min_value = np.min(data_minmax)
        max_value = np.max(data_minmax)
        print(min_value, max_value)
        for idx, data in enumerate(raw.get_data()):
            # Initialize a list to store the normalized values
            normalized_data = []

            # Normalize the values in the dataset
            for value in data:
                normalized_value = (value - min_value) / (max_value - min_value)
                normalized_data.append(normalized_value)

            smoothed_data.append(normalized_data)

    if norm == 1:
        for_smoothing = smoothed_data
    else:
        for_smoothing = raw.get_data()

    if smoothing == 1:
        for idx, data in enumerate(for_smoothing):
            # Glättungsfenster erstellen
            weights = np.repeat(1.0, float(smoothing)) / float(smoothing)

            # Running average berechnen
            smooth = np.convolve(data, weights, 'same')
            for_smoothing[idx] = smooth

    print("finished")
    return raw, for_smoothing


def load_np_array_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save_np_array_pickle(filename, array):
    with open(filename, 'wb') as f:
        pickle.dump(array, f)


def shift_data(data, data2, step):
    data_arr = np.array(data)
    real_data = data_arr
    data_arr2 = np.array(data2)
    t = 0
    correlation_results = []  # Create array to store cluster results

    while t == 0:


        copy_data = np.copy(data_arr)
        copy_data[step:] = data_arr[:-step]  # Verschiebe Daten um step nach rechts
        copy_data[:step] = data_arr[-step:]  # füge die ersten step Daten an das Ende an
        data_arr = copy_data

        correlation = np.corrcoef(data_arr, data_arr2)[0, 1]  # Calculate cross cluster
        correlation_results.append(correlation)  # add the cluster to the results array

        if np.array_equal(data_arr, real_data):
            t = 1

    return correlation_results


def find_highest_peak(data):
    data_array = np.array(data)
    highest_peak_index = np.argmax(data_array)
    highest_peak_value = data_array[highest_peak_index]
    return highest_peak_index,highest_peak_value


def correlate_epochs(window, epochs, threshold):
    correlation_idx = []
    correlation_data = []
    for idx, epoch in enumerate(epochs):
        if len(window) == len(epoch):
            # shift the window
            s_data = shift_data(window, epoch, 1)
            # find peak info
            peak_index, peak_value = find_highest_peak(s_data)

            if peak_value > threshold:
                correlation_idx.append(idx)
                correlation_data.append(epoch)

    return correlation_idx, correlation_data


def correlate_matrix(window, windows_dict):
    correlation_data = []
    for idx, data in enumerate(windows_dict):
        if len(window) == len(data['data']):
            # shift the window

            s_data = shift_data(window, data['data'], 1)
            # find peak info
            peak_index, peak_value = find_highest_peak(s_data)
            correlation_data.append(peak_value)


    return correlation_data


def raw_filter(raw, ica, notch, lowpass, highpass):
    filtered_raw = raw.copy()
    if ica:
        # Data preprocessing
        try:
            # Create ICA object
            ica = ICA(n_components=10, random_state=42)

            # Fit ICA on raw data
            ica.fit(filtered_raw)

            # Identify and remove undesirable components
            ica.exclude = [ica.labels_['eye_blink', 'muscle_activity', 'heartbeat']]

            # Apply ICA to raw data
            ica.apply(filtered_raw)

        except KeyError:
            pass

    if notch:
        notch_frequencies = [50, 60]

        # Anwendung des Notch-Filters
        filtered_raw = filtered_raw.notch_filter(notch_frequencies)

    if lowpass and highpass:
        # Definieren Sie die Filtergrenzen (z.B. 0,5 - 30 Hz)
        l_freq = float(lowpass)
        h_freq = float(highpass)
        # Anwenden des Butterworth-Filters
        filtered_raw.filter(l_freq, h_freq, fir_design='firwin')

    return filtered_raw


def generate_random_noise(time, min_amp, max_amp, n_signals, prob_min=0.3, prob_max=0.3):
    output_list = []

    for _ in range(n_signals):
        output = np.zeros(time)

        for i in range(time):
            random_num = np.random.rand()

            if random_num <= prob_min:
                output[i] = np.random.uniform(min_amp - 0.05, min_amp + 0.05)
            elif random_num <= prob_min + prob_max:
                output[i] = np.random.uniform(max_amp - 0.05, max_amp + 0.05)
            else:
                output[i] = np.random.uniform(min_amp, max_amp)

        output_list.append(output)

    return output_list


import numpy as np

def detect_peak(data_list, threshold):
    detected_peaks = []
    counter = 0
    c = 0

    for i, entry in enumerate(data_list):
        data = np.array(entry['data'])
        max_peak = data.max()

        # Berechnen Sie time_step und start_time für jeden Eintrag
        start_time, end_time = map(float, entry['time'].split('-'))
        time_step = (end_time - start_time) / len(data)

        if max_peak > threshold:
            peak_time_index = np.argmax(data)
            peak_time = start_time + peak_time_index * time_step

            # Prüfen, ob der Peak bereits erkannt wurde
            duplicate_peak = False

            # Vergleichen Sie den aktuellen Peak mit den erkannten Peaks innerhalb des erlaubten Zeitfensters
            for detected_peak in detected_peaks:
                if abs(peak_time - detected_peak['time']) < (1 / 500):
                    duplicate_peak = True
                    print(f"Doppelter Peak erkannt: Zeit: {peak_time}, Höhe des Peaks: {max_peak}, Indizes: {detected_peak['index']}, {i}")
                    counter += 1
                    break

            if not duplicate_peak:
                detected_peaks.append({'time': peak_time, 'peak': max_peak, 'index': i})  # Speichern Sie den Index zusammen mit der Peak-Zeit und dem Peak-Wert
                print(f"Zeit: {peak_time}, Höhe des Peaks: {max_peak}")
                c += 1

    print(counter)
    print(c)

def get_timepoint_for_index(data, index):
    time_range = data['time']
    data_points = data['data']
    start_time, end_time = map(float, time_range.split('-'))

    if index < 0 or index >= len(data_points):
        raise IndexError("Index out of range")

    time_step = (end_time - start_time) / len(data_points)
    time_point = start_time + index * time_step

    return time_point


def normalize_epochs(win):
    # Extrahieren der Daten aus jedem Objekt in der Liste
    data_list = [obj['data'] for obj in win]

    # Konvertieren der Daten in ein 1D-Array
    data_array = np.concatenate(data_list)

    # Berechnen des Maximums und Minimums der Daten
    max_value = np.max(data_array)
    min_value = np.min(data_array)

    for idx, data in enumerate(win):
        normalized_data = []
        for value in data['data']:
            normalized_value = (value - min_value) / (max_value - min_value)
            normalized_data.append(normalized_value)
        win[idx]['data'] = normalized_data
    return win


def augment_data_list(data_list, count):

    def add_noise(d, noise_factor=0.05):
        noise = np.random.normal(0, noise_factor, len(d))
        return d + noise

    def scale_data(d, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return d * scale

    def time_shift(d, shift_range=(-10, 10)):
        shift = np.random.randint(shift_range[0], shift_range[1])
        return np.roll(d, shift)
    # Test
    augmented_data_list = []
    for _ in range(count):
        # Wähle zufällig eine Datenreihe aus der Liste
        data_idx = np.random.randint(0, len(data_list))
        data = data_list[data_idx]['data']

        choice = np.random.randint(0, 7)
        if choice == 0:
            new_data = add_noise(data)
        elif choice == 1:
            new_data = scale_data(data)
        elif choice == 2:
            new_data = time_shift(data)
        elif choice == 3:
            new_data = add_noise(scale_data(data))
        elif choice == 4:
            new_data = add_noise(time_shift(data))
        elif choice == 5:
            new_data = scale_data(time_shift(data))
        else:
            new_data = add_noise(scale_data(time_shift(data)))
        augmented_data_list.append(new_data)
    return augmented_data_list



if __name__ == "__main__":
    epoch_dict = load_np_array_pickle('../files/epochs_files/epoch_check.pickle')
    epoch_dict = normalize_epochs(epoch_dict)
    cleared_epochs, deleted_epochs = clear_flatlines(epoch_dict, 0.02)
    print(len(cleared_epochs))
    show_plots(cleared_epochs)