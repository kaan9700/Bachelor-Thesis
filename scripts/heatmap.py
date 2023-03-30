from functions import load_np_array_pickle
import numpy as np
from collections import defaultdict


def count_unique_times(data_list, tolerance=0.1):
    unique_times = []

    for item in data_list:
        time_value = item['time']
        is_unique = True

        for unique_time in unique_times:
            if abs(time_value - unique_time) <= tolerance:
                is_unique = False
                break

        if is_unique:
            unique_times.append(time_value)

    return len(unique_times)


data_list = load_np_array_pickle('../files/predicted_data.pickle')



import numpy as np
from collections import defaultdict

# Gruppieren Sie die Daten nach 'ch'-Werten.
grouped_data = defaultdict(list)
for data in data_list:
    grouped_data[data['ch']].append(data)

# Sortieren Sie die gruppierten Daten nach 'time'-Werten.
for ch, data in grouped_data.items():
    grouped_data[ch] = sorted(data, key=lambda x: x['time'])

# Berechnen Sie die einzigartigen 'time'-Werte und ordnen Sie sie in Spalten an.
unique_times = sorted(set(round(item['time'], 1) for item in data_list))

# Erstellen Sie eine Matrix und füllen Sie sie entsprechend mit 'time'-Werten oder leeren Werten.
n_rows = len(grouped_data)
n_cols = len(unique_times)
matrix = np.empty((n_rows, n_cols), dtype=object)

for row_idx, (ch, data) in enumerate(grouped_data.items()):
    for col_idx, time in enumerate(unique_times):
        time_values = [d['time'] for d in data if abs(d['time'] - time) <= 0.1]
        if time_values:
            matrix[row_idx, col_idx] = time_values[0]
        else:
            matrix[row_idx, col_idx] = None

print(matrix.shape)