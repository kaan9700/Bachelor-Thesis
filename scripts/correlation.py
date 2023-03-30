import numpy as np
import scipy as sp
import time
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm

def load_np_array_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def pool_initializer(result):
    i, j, new_matr = result
    corr_matrix[i, j] = new_matr[0,1]
    corr_matrix[j, i] = new_matr[1,0]

def calc_corr(args):
    i, j = args
    sig1 = data_dict[i]['data']
    sig2 = data_dict[j]['data']
    nsig1 = sig1 - np.mean(sig1)
    nsig2 = sig2 - np.mean(sig2)
    corr = sp.signal.correlate(nsig1, nsig2, mode='same', method='fft')
    corr /= (len(sig2) * np.std(sig1) * np.std(sig2))
    max_corr = find_highest_peak(corr)
    temp_matrix = np.zeros((2,2))
    temp_matrix[0,1] = max_corr[1]
    temp_matrix[1,0] = max_corr[1]

    return i, j, temp_matrix

def find_highest_peak(data):
    data_array = np.array(data)
    highest_peak_index = np.argmax(data_array)
    highest_peak_value = data_array[highest_peak_index]
    return highest_peak_index, highest_peak_value


data_dict = load_np_array_pickle('../files/windows_files/final_windows.pickle')
# data_dict = load_np_array_pickle('../files/epochs_files/final_epochs.pickle')

if __name__ == '__main__':
    print("Starte Kreuzkorrelation...")
    print(len(data_dict))
    start = time.perf_counter()
    corr_len = len(data_dict)
    corr_matrix = np.zeros((corr_len, corr_len))
    status_percentage = int(corr_len / 100) if corr_len > 100 else 1

    with Pool(processes=cpu_count()) as p:
        results = p.imap_unordered(calc_corr, [(i, j) for i in range(corr_len) for j in range(i+1, corr_len)])
        for result in tqdm(results, total=int(corr_len * (corr_len - 1) / 2), desc="Fortschritt"):
            pool_initializer(result)

    np.fill_diagonal(corr_matrix, 1.)
    end = time.perf_counter()
    print('\nAufgebrachte Zeit:  {:.2f} s'.format(end - start, 2))
    print(corr_matrix)
    with open("../files/correlation_files/correlation_windows.pickle", 'wb') as f:
        pickle.dump(corr_matrix, f)
