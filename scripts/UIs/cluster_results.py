import tkinter as tk
from tkinter import *
import sys
sys.path.insert(0, "..")
from functions import load_np_array_pickle
import matplotlib.backends.backend_tkagg as backend_tkagg
import matplotlib.pyplot as plt
import numpy as np


class ClusterGUI(Tk):
    def __init__(self):
        super().__init__()
        self.view_1 = startView(self, clusters)

class startView:
    def __init__(self, master, clusters):
        self.master = master
        self.clusters = clusters
        self.master.title("Cluster List")
        self.label = tk.Label(master, text="Cluster List")
        self.label.pack()
        self.cluster_list = tk.Listbox(master)
        self.clusts = self.count_occurrences(self.clusters)
        for clust in self.clusts:
            self.cluster_list.insert('end', "Cluster {} ({} Elements)".format(clust[0], clust[1]))
        self.cluster_list.pack()
        self.cluster_list.bind("<Double-Button-1>", self.show_cluster)

    def show_cluster(self, event):
        widget = event.widget
        index = widget.curselection()[0]
        cluster = self.clusts[index]
        self.go_to_next_view(cluster, self.clusters)

    def count_occurrences(self, arrs):
        count_dict = {}
        for num in arrs:
            if num in count_dict:
                count_dict[num] += 1
            else:
                count_dict[num] = 1
        return [(num, count) for num, count in count_dict.items()]

    def go_to_next_view(self, cluster, clusters):
        for widget in self.master.winfo_children():
            widget.destroy()  # destroy all widgets in the current window

        view_2 = Cluster_info(self.master, cluster, clusters)
        view_2.pack(fill='both', expand=True)




class Cluster_info(Frame):
    def __init__(self, parent, cluster, clusters, *args, **kwargs):

        super().__init__(parent, *args, **kwargs)

        self.epoch_dict = load_np_array_pickle('../../files/windows_files/final_windows.pickle')
        print(len(self.epoch_dict))
        print(len(clusters))
        l1 = Label(self, text="Cluster: {}".format(cluster[0]))
        l1.pack(side='top', pady=(25, 10))

        l2 = tk.Label(self, text="Number of Elements: {}".format(cluster[1]))
        l2.pack(side='top')

        self.button = Button(self, text="Back", command=self.go_back)
        self.button.pack(side='bottom', pady=(0, 30))
        self.windows = self.get_windows(clusters, cluster[0])

        # Erstellen Sie ein Figure-Objekt und eine Axes-Instanz
        self.selectedWindow = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.shown_plot = self.windows[self.selectedWindow]['data']
        self.ax.plot(self.shown_plot)

        # Erstellen Sie eine FigureCanvasTkAgg-Instanz
        self.canvas = backend_tkagg.FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().place(relx=0.2, rely=0.05)
        #self.canvas.get_tk_widget().bind('<Button-1>', lambda event: self.show_plot(self.shown_plot, sfreq))
        self.canvas.get_tk_widget().configure(width=480, height=250)

        self.next = Button(self, text='Next Window', command=self.next_window)
        self.next.place(relx=0.5, rely=0.6)

        self.previous = Button(self, text='Previous Window', command=self.previous_window)
        self.previous.place(relx=0.3, rely=0.6)

    def go_back(self):
        self.pack_forget() # destroy view 2
        view_1 = startView(self.master, clusters)


    def get_windows(self, clusters, clusterNr):
        indizes = np.where(clusters == clusterNr)
        print('indizes: ', indizes[0][0])
        clust_epochs = [self.epoch_dict[el] for el in indizes[0]]
        return clust_epochs

    def next_window(self):
        self.selectedWindow += 1
        self.ax.clear()

        self.ax.plot(self.windows[self.selectedWindow]['data'])
        self.fig = plt.figure(figsize=(10, 5))
        # Aktualisieren Sie das Canvas
        self.canvas.draw()
        self.canvas.get_tk_widget().configure(width=480, height=250)


    def previous_window(self):
        self.selectedWindow -= 1
        self.ax.clear()

        self.ax.plot(self.windows[self.selectedWindow]['data'])
        self.fig = plt.figure(figsize=(10, 5))
        # Aktualisieren Sie das Canvas
        self.canvas.draw()
        self.canvas.get_tk_widget().configure(width=480, height=250)





if __name__ == "__main__":
    w_e = input('windows oder epochs')
    if w_e == 'w':
        prf = 'windows'
    else:
        prf = 'epochs'

    cluster_type = input("Bitte wählen Sie Cluster-Algorithmus aus: \n1 = dbscan\n2 = hierachical\n3 = kmeans\n4 = "
                         "affinity propagation\n")
    if cluster_type == "1":
        clusters = load_np_array_pickle('../../files/cluster_files/dbscan_'+prf+'.pickle')
    if cluster_type == "2":
        clusters = load_np_array_pickle('../../files/cluster_files/hierachical_'+prf+'.pickle')
    if cluster_type == "3":
        clusters = load_np_array_pickle('../../files/cluster_files/kmeans_'+prf+'.pickle')
    if cluster_type == "4":
        clusters = load_np_array_pickle('../../files/cluster_files/ap.pickle')

    app = ClusterGUI()
    app.title('EEG Clustering')

    # Setze die Höhe und Breite des Fensters und verhindere, dass es verändert wird
    app.geometry("800x450")
    app.resizable(False, False)
    app.mainloop()
