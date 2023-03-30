import numpy as np

from functions import create_windows, eeg_filter, load_np_array_pickle, show_plots, detect_peak, save_np_array_pickle, get_timepoint_for_index
import os
import mne
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def select_file(filename, directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    count = 0
    for file in file_list:
        count += file.count(filename)
    if count == 3:
        return True
    else:
        return False


def create_windows_matrix(win):
    model = load_model('./NeuralNet/models/ffnn.h5')

    win_matrix = np.zeros((len(win), len(win[0])))

    batch_size = 64  # Sie können die Batch-Größe anpassen, um die beste Leistung für Ihr System zu erzielen

    for i, channel in enumerate(win):
        num_batches = (len(channel) + batch_size - 1) // batch_size  # Berechnen der Anzahl der Batches

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(channel))
            batch_data = np.stack(channel[start_idx:end_idx], axis=0)  # Erstellen eines Batches aus den Daten

            # Vorhersagen für den gesamten Batch treffen
            batch_preds = model.predict(batch_data)
            batch_preds_argmax = batch_preds.argmax(axis=1)

            for j, pred in enumerate(batch_preds_argmax, start=start_idx):
                if pred != 4:
                    win_matrix[i][j] = 1

    # Zählen der Einsen in der Matrix
    count_ones = np.sum(win_matrix == 1)

    print("Anzahl der Einsen:", count_ones)
    with open('../files/prediction_matrix.pickle', 'wb') as f:
        pickle.dump(win_matrix, f)



def find_multi_channel_peaks(matrix, threshold=1):
    num_channels, num_windows = matrix.shape
    multi_channel_peaks = []

    for window_idx in range(num_windows):
        channels_with_peak = []

        for channel_idx in range(num_channels):
            if matrix[channel_idx, window_idx] == 1:
                channels_with_peak.append(channel_idx)

        # Wenn mehr als ein Kanal einen Ausschlag im aktuellen Fenster hat
        if len(channels_with_peak) > 1:
            multi_channel_peaks.append((window_idx, channels_with_peak))

        # Prüfung auf benachbarte Fenster unter Berücksichtigung des Schwellenwerts
        for offset in range(1, threshold + 1):
            if window_idx + offset < num_windows:
                adjacent_channels_with_peak = []

                for channel_idx in range(num_channels):
                    if matrix[channel_idx, window_idx + offset] == 1:
                        adjacent_channels_with_peak.append(channel_idx)

                common_channels = set(channels_with_peak) & set(adjacent_channels_with_peak)
                if common_channels:
                    multi_channel_peaks.append((window_idx, window_idx + offset, list(common_channels)))

    return multi_channel_peaks


def find_data_points_at_time(windows, target_time):
    data_points = []

    for window in windows:
        time_range = window['time']
        data_points_list = window['data']

        start_time, end_time = map(float, time_range.split('-'))

        if target_time >= start_time and target_time <= end_time:
            time_step = (end_time - start_time) / len(data_points_list)
            index = round((target_time - start_time) / time_step)

            if index >= 0 and index < len(data_points_list):
                data_points.append(data_points_list[index])

    return data_points


if __name__ == '__main__':
    # raw_name = input('Gebe den Namen der Datei ein (ohne Dateiformat)')

    if not os.path.isfile('../files/prediction_matrix.pickle'):
        if not os.path.isfile('fs.pickle'):
            raw_name = 'Control1415'
            filename = ''
            root = '../files/'
            if select_file(raw_name, root):
                filename = raw_name

            raw = mne.io.read_raw_brainvision(root + filename + '.vhdr', preload=True)
            ica = 1
            notch = 1
            low_pass = 8 / (raw.info['sfreq'] / 2)
            high_pass = 30 / (raw.info['sfreq'] / 2)
            filtered_raw, fs = eeg_filter(ica=ica, notch=notch, low=low_pass, high=high_pass, norm=1, smoothing=0, filename='../'+root+filename+'.vhdr')
            fs = fs[:-1]

            # Speichern der Variable fs in der Datei fs.pickle
            save_np_array_pickle('fs.pickle', fs)
            print('saved')
        else:
            with open('fs.pickle', 'rb') as f:
                fs = pickle.load(f)
        print('next step')
        windows = create_windows(0.5, 301, fs[0], 500)

        i = 0
        for window in windows[0]:
            if np.array(window['data']).max() > 0.45:
                i += 1
                time = get_timepoint_for_index(window, np.array(window['data']).argmax())
                print(f'Zeitpunkt: {time}')
        #
        print(find_data_points_at_time(windows[0], 77.1))

    else:
        windows_matrix = load_np_array_pickle('../files/prediction_matrix.pickle')
        multi_channel_peaks = find_multi_channel_peaks(windows_matrix, threshold=1)
        print("Mehrfachkanal-Ausschläge gefunden bei:", multi_channel_peaks)
        print(len(multi_channel_peaks))