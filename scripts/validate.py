import mne
import numpy as np
from tensorflow.keras.models import load_model
from functions import create_windows, create_window_dict, normalize_epochs
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt

filename = input('Geben Sie den Namen der Validierungsdatei an')
raw = mne.io.read_raw_brainvision('../files/validate_files/' + filename, preload=True)


# Signalverarbeitung
lowpass = 0.5
highpass = 30

# Notch Frequenzen
notch_frequencies = [50, 60]

filtered_raw = raw.copy()

# Anwendung des Notch-Filters
filtered_raw.notch_filter(notch_frequencies)

l_freq = float(lowpass)
h_freq = float(highpass)
# Definieren Sie die Filtergrenzen
l_freq = float(l_freq)
h_freq = float(h_freq)
# Anwenden des Butterworth-Filters
filtered_raw.filter(l_freq, h_freq, fir_design='firwin')


# ICA NOCH MACHEN


# Fenster erstellen:
windows = create_windows(0.6666666, 300, filtered_raw.get_data()[:-1], 500)
print("Anzahl der erstellten fenster: (", len(windows), len(windows[0]), ")")

windows_dict = create_window_dict(windows, filtered_raw)
print(len(windows_dict))

#Normalisieren
normalized_windows_dict = normalize_epochs(windows_dict)


# Modell anwenden
model = load_model('./NeuralNet/models/ffnn.h5')

batch_size = 512
prediction_list = []

# Anzahl der Batches berechnen
num_batches = int(np.ceil(len(normalized_windows_dict) / batch_size))

for i in range(num_batches):
    # Start- und Endindex für den aktuellen Batch extrahieren
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(normalized_windows_dict))

    # Aktuellen Batch aus den Daten extrahieren und die Dimension erweitern
    batch_data = [np.expand_dims(data['data'], axis=0) for data in normalized_windows_dict[start_idx:end_idx]]
    batch_data = np.concatenate(batch_data, axis=0)

    # Vorhersagen für den aktuellen Batch durchführen
    batch_preds = model.predict(batch_data)

    # Klassenlabels aus den Wahrscheinlichkeiten extrahieren und der Vorhersageliste hinzufügen
    batch_pred_labels = np.argmax(batch_preds, axis=1)
    prediction_list.extend(batch_pred_labels)


def get_time_at_index(time_range, index, data_length):
    start_time, end_time = time_range
    time_step = (end_time - start_time) / data_length

    return start_time + (index * time_step)


def is_within_buffer(existing_peaks, new_peak, buffer):
    for peak in existing_peaks:
        if peak['ch'] == new_peak['ch'] and abs(peak['time'] - new_peak['time']) <= buffer:
            return True
    return False

buffer = 1  # Puffer in Sekunden, passen Sie diesen Wert entsprechend an
peaks = []
peaks_data = []
for idx, data in enumerate(normalized_windows_dict):
    if prediction_list[idx] == 1:
        meta = {'time': get_time_at_index(data['time'], np.array(data['data']).argmax(), 300), 'ch': data['channel']}

        if not is_within_buffer(peaks, meta, buffer):
            peaks.append(meta)
            peaks_data.append(data['data'])
        else:
            for peak in peaks:
                if peak['ch'] == meta['ch'] and abs(peak['time'] - meta['time']) <= buffer:
                    if meta['time'] < peak['time']:
                        peak['time'] = meta['time']
                    break


predicted_data = []
for i, data in enumerate(peaks):
    predicted_data.append({'time': data['time'], 'ch': data['ch'], 'data': peaks_data[i]})


# Gruppieren Sie die Daten nach 'ch'-Werten.
grouped_data = defaultdict(list)
for data in predicted_data:
    grouped_data[data['ch']].append(data)


time_values = [data['time'] for data in predicted_data]
min_time = min(time_values)
max_time = max(time_values)
# Ranges erstellen
step = 0.1

ranges = []
current_time = min_time - (step/2)

while current_time <= max_time:
    lower_bound = round(current_time, 2)
    upper_bound = round(current_time + step, 2)
    time_range = (lower_bound, upper_bound)
    ranges.append(time_range)
    current_time += step

amount_cols = len(ranges)

matrix = np.zeros((len(filtered_raw.ch_names)-1, amount_cols))
print(matrix.shape)


def find_range(ranges, x):
    for index, (lower_bound, upper_bound) in enumerate(ranges):
        if lower_bound <= x < upper_bound:
            return index
    return None


for i, ch in enumerate(filtered_raw.ch_names[:-1]):
    for el in grouped_data[ch]:
        range_index = find_range(ranges, el['time'])

        # Berechnen des Integrals mit der Trapezregel
        integral = np.trapz(el['data'])

        # Eintragen des Integrals in die Matrix
        matrix[i][range_index] = integral

# Index der Spalten, die entfernt werden müssen
cols_to_remove = []

# Schleife über alle Spalten des Arrays
for j in range(matrix.shape[1]):
    # Überprüfen, ob alle Elemente in der aktuellen Spalte gleich Null sind
    if np.count_nonzero(matrix[:,j]) == 0:
        cols_to_remove.append(j)

# Spalten entfernen und Indexe in ranges entfernen
matrix = np.delete(matrix, cols_to_remove, axis=1)
ranges = [ranges[i] for i in range(len(ranges)) if i not in cols_to_remove]

final_matrix = np.where(matrix ==0, 0, matrix)
print(final_matrix.shape)

# Berechnen Sie die Kosinusähnlichkeit für die Zeilen der Matrix
similarity_matrix = cosine_similarity(final_matrix)
print(similarity_matrix.shape)
# Erstellen Sie eine Heatmap der Kosinusähnlichkeitsmatrix
sns.heatmap(similarity_matrix, cmap="coolwarm")

plt.title("Kosinusähnlichkeit zwischen Zeilen der Matrix")
plt.show()

# Berechnen Sie die Distanzmatrix (1 - Kosinusähnlichkeit)
distance_matrix = 1 - similarity_matrix
print(distance_matrix.shape)
# Führen Sie hierarchisches Clustering durch
clusters = linkage(distance_matrix, method='complete')
print(clusters.shape)

# Zeichnen Sie das Dendrogramm
# Schwellenwert für Clusterbildung festlegen und Clusterzuordnungen abrufen
threshold = 0.5  # Schwellenwert anpassen, um die Anzahl der Cluster zu steuern
cluster_assignments = fcluster(clusters, threshold, criterion='distance')



print((cluster_assignments))


# Sortiere die Matrix entsprechend der Clusterzuordnung
sorted_indices = np.argsort(cluster_assignments)
sorted_matrix = matrix[sorted_indices]

subset_matrix = sorted_matrix[:, :20]

# Erstelle die Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(subset_matrix, cmap="YlOrBr")
plt.xlabel("Peaks")
plt.ylabel("Segmente")
plt.title("Heatmap der Kanäle und Segmente")
plt.show()