from tkinter import *
import tkinter.filedialog as fd
import os
import shutil
import sys
sys.path.insert(0, "../")
from functions import get_information, show_all_plots, create_windows, eeg_filter
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as backend_tkagg
from tkinter import ttk
import pickle

class EEG_GUI(Tk):
    def __init__(self):
        super().__init__()
        self.view_1 = Upload_data(self)
        self.view_1.pack(fill='both', expand=True)


class Upload_data(Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.files = ()

        # Eingabe-Label und -Eingabefeld hinzufügen
        label = Label(self, text='Upload EEG data for processing')
        label.pack(side='top', pady=(25, 10))
        upload_button = Button(self, text='Select file', command=self.edit_upload_file)
        upload_button.pack(side='top')

        self.text_field = Text(self, background='white', fg='black', width=50, height=10)
        self.text_field.pack_forget()

        self.continue_button = Button(self, text='Next', command=self.upload_files)
        self.continue_button.pack_forget()

        button_old_data = Button(self, text="Continue with existing data", command=self.go_to_next_view)
        button_old_data.pack(side='bottom', pady=(0, 30))

    def go_to_next_view(self):
        self.pack_forget()  # destroy view 1
        view_2 = Select_data(self.master)
        view_2.pack(fill='both', expand=True)

    def edit_upload_file(self):
        files = fd.askopenfilenames(parent=self, title="Choose a file")
        if files:
            namen = get_filenames(files)
            self.text_field.pack()

            for name in namen:
                if name not in self.text_field.get(1.0, 'end'):
                    self.text_field.insert('end', name)
                    self.text_field.insert('end', '\n')

            self.continue_button.pack(side='top')
            self.files += files

    def upload_files(self):
        # Ermitteln Sie das Verzeichnis, in dem das Skript liegt
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Kopieren Sie die ausgewählten Dateien in das Verzeichnis
        for file in self.files:
            shutil.copy(file, script_dir + '/files/eeg_files/')
        self.text_field.delete(1.0, 'end')
        self.text_field.pack_forget()
        self.continue_button.pack_forget()
        self.go_to_next_view()


class Select_data(Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.label = Label(self, text="Select one of the available files")
        self.label.pack(side='top',  pady=(25, 30))
        self.files = self.get_uploaded_files()
        self.listbox = Listbox(self)
        for file in self.files:
            self.listbox.insert(END, file)
        self.listbox.pack(side='top')
        self.selected_file = ""
        self.listbox.bind('<<ListboxSelect>>', self.data_details)

        self.button = Button(self, text="Back", command=self.go_back)
        self.button.pack(side='bottom', pady=(0, 30))

        self.next_button = Button(self, text='Next', command=self.go_to_next_view)
        self.next_button.pack_forget()

    def data_details(self, event):
        try:
            selection = event.widget.curselection()[0]
        except IndexError:
            return
        self.next_button.pack(side='top', pady=(10, 0))
        self.selected_file = self.files[selection]

    def go_back(self):
        self.pack_forget() # destroy view 2
        view_1 = Upload_data(self.master)
        view_1.pack(fill='both', expand=True)

    def go_to_next_view(self):
        self.pack_forget()  # destroy view 1
        view_2 = Data_Details(self.master, filename=self.selected_file)
        view_2.pack(fill='both', expand=True)

    def get_uploaded_files(self):
        dir = '../../files/eeg_files/'
        files = []
        for file in os.listdir(dir):
            if file.endswith('.eeg') or file.endswith('mat'):
                files.append(file)
        return files


class Data_Details(Frame):
    def __init__(self, parent, filename, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Variablen
        sfreq, channels, scan_dur, raw = get_information(filename)
        raw_data = raw.get_data()
        self.raw_filtered = 0
        self.smoothed_plot = 0


        # Erstellen der drei Frames
        frame1 = Frame(self)
        separator1 = ttk.Separator(self, orient='vertical')
        separator1.place(relx=0.25, rely=0, relwidth=0.0014, relheight=1)
        frame2 = Frame(self)
        separator2 = ttk.Separator(self, orient='horizontal')
        separator2.place(relx=0, rely=0.516, relwidth=1, relheight=0.002)
        frame3 = Frame(self)


        # Verwenden der grid-Methode, um die Frames in einem 2x3-Raster anzuordnen
        frame1.grid(row=0, column=0, columnspan=1, sticky='nsew')
        frame2.grid(row=0, column=1, columnspan=4, sticky='nsew')
        frame3.grid(row=1, column=0, columnspan=5, sticky='nsew')

        # Verwenden Sie die weight-Option, um die Größe der Frames in der ersten Zeile zu steuern
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=4)

        # Widgets erstellen
        # Information-Frame
        inf_label = Label(frame1, text='Information', font="font_bold")
        inf_label.pack(side='top')

        name_label = Label(frame1, text="Name:    "+filename)
        name_label.place(relx=0, rely=0.2)

        freq_label = Label(frame1, text="Scan Duration:    " + str(scan_dur) + 's')
        freq_label.place(relx=0, rely=0.4)

        freq_label = Label(frame1, text="Sampling-Rate:    " + str(sfreq) + 'Hz')
        freq_label.place(relx=0, rely=0.6)

        channel_label = Label(frame1, text="Number of channels:    " + str(len(channels)))
        channel_label.place(relx=0, rely=0.8)

        # Plot-Frame
        used_channel_label = Label(frame2, text="Channel: ")
        used_channel_label.place(relx=0.0, rely=0.1)

        # Erstellen Sie ein ttk.Combobox-Widget
        self.combo = ttk.Combobox(frame2, values=channels, state="readonly")
        self.combo.set(channels[0])
        self.combo.place(relx=0, rely=0.2)
        self.combo.bind('<<ComboboxSelected>>', lambda event: self.change_plot(timestamps, self.determine_raw_data(raw_data), channels, sfreq))

        # Anzeige aller Plots
        all_plots = Button(frame2, text='Show all Plots', command=lambda: show_all_plots(self.determine_raw(raw)))
        all_plots.place(relx=0, rely=0.4)

        # Erstellen Sie ein Figure-Objekt und eine Axes-Instanz
        self.fig, self.ax = plt.subplots(figsize=(10,5))
        timestamps = [i / sfreq for i in range(len(raw_data[0]))]
        self.shown_plot = raw_data[0]
        self.ax.plot(self.shown_plot)
        self.ax.set_xlabel("Time in s")
        self.ax.set_ylabel("Brain activity in µV")

        # Erstellen Sie eine FigureCanvasTkAgg-Instanz
        self.canvas = backend_tkagg.FigureCanvasTkAgg(self.fig, master=frame2)
        self.canvas.get_tk_widget().place(relx=0.3, rely=0.05)
        self.canvas.get_tk_widget().bind('<Button-1>', lambda event: self.show_plot(self.shown_plot, sfreq))
        self.canvas.get_tk_widget().configure(width=480, height=250)


        #Data Processing Frame
        epochs_label = Label(frame3, text="Create Epochs")
        epochs_label.place(relx=0.1, rely=0)

        epochs_overlap_label = Label(frame3, text="Overlap: ")
        epochs_overlap_label.place(relx=0, rely=0.2)
        self.epochs_overlap_text = Text(frame3, bg='grey', width=10, height=1)
        self.epochs_overlap_text.config(font=("Arial", 18))
        self.epochs_overlap_text.place(relx=0.1, rely=0.2)
        self.epochs_overlap_text.bind('<KeyRelease>', self.check_values)

        epochs_window_label = Label(frame3, text="Window size: ")
        epochs_window_label.place(relx=0, rely=0.35)
        self.epochs_window_text = Text(frame3, bg='grey', width=10, height=1)
        self.epochs_window_text.config(font=("Arial", 18))
        self.epochs_window_text.place(relx=0.1, rely=0.35)
        self.epochs_window_text.bind('<KeyRelease>', self.check_values)

        self.continue_epochs_button = Button(frame3, text='Create Epochs', command=lambda: self.open_epochs_view(
            self.epochs_overlap_text.get(1.0, 'end'), self.epochs_window_text.get(1.0, 'end'), self.determine_raw_data(raw_data), filename, channels, sfreq, self.determine_raw_data(raw_data)))
        self.continue_epochs_button.pack_forget()


        self.ica_status = IntVar()
        self.notch_status = IntVar()
        self.normalize_status = IntVar()
        self.high_status = 0
        self.low_status = 0
        self.smoothing_status = 0

        filter_label = Label(frame3, text='Filter')
        filter_label.place(relx=0.7, rely=0)

        self.check_ica = Checkbutton(frame3, variable=self.ica_status)
        self.check_ica.place(relx=0.6, rely=0.2)
        ica_label = Label(frame3, text="Independent Component Analysis")
        ica_label.place(relx=0.65, rely=0.2)

        self.check_notch = Checkbutton(frame3, variable=self.notch_status)
        self.check_notch.place(relx=0.6, rely=0.3)
        notch_label = Label(frame3, text="Notch Filter")
        notch_label.place(relx=0.65, rely=0.3)


        low_high_filter_label = Label(frame3, text="Set Low- & Highpass Filter")
        low_high_filter_label.place(relx=0.6, rely=0.4)
        self.low_filter = Text(frame3, bg='grey', width=5, height=1)
        self.low_filter.config(font=("Arial", 18))
        self.low_filter.place(relx=0.8, rely=0.4)
        self.low_filter.bind('<KeyRelease>', lambda event: self.high_low_filter_set(event, 'l'))

        self.high_pass = Text(frame3, bg='grey', width=5, height=1)
        self.high_pass.config(font=("Arial", 18))
        self.high_pass.place(relx=0.9, rely=0.4)
        self.high_pass.bind('<KeyRelease>', lambda event: self.high_low_filter_set(event, 'h'))

        self.check_normalize = Checkbutton(frame3, variable=self.normalize_status)
        self.check_normalize.place(relx=0.6, rely=0.5)
        ica_label = Label(frame3, text="Normalize Data")
        ica_label.place(relx=0.65, rely=0.5)

        smoothing_label = Label(frame3, text="Set Smoothing Window")
        smoothing_label.place(relx=0.6, rely=0.6)
        self.smoothing = Text(frame3, bg='grey', width=5, height=1)
        self.smoothing.config(font=("Arial", 18))
        self.smoothing.place(relx=0.8, rely=0.6)
        self.smoothing.bind('<KeyRelease>', lambda event: self.high_low_filter_set(event, 's'))


        self.filter_update_button = Button(frame3, text='Apply', command=lambda: self.apply_filters(raw, channels, sfreq, timestamps, filename))
        self.filter_update_button.place(relx=0.6, rely=0.8)

    def check_values(self, event):
        if self.epochs_overlap_text.get(1.0, 'end').strip() and self.epochs_window_text.get(1.0, 'end').strip():
            self.continue_epochs_button.place(relx=0, rely=0.5)
        else:
            self.continue_epochs_button.place(relx=-0.5)

    def go_back(self):
        self.pack_forget() # destroy view 2
        view_1 = Select_data(self.master)
        view_1.pack(fill='both', expand=True)

    def show_plot(self, val, freq):
        plt.close('all')
        timestamps = [i / freq for i in range(len(val))]
        plt.plot(val)
        plt.show()

    def change_plot(self, timestamps, raw_data, channels, sfreq):
        selected = self.combo.get()
        index_channel = channels.index(selected)

        self.shown_plot= raw_data[index_channel]
        self.ax.clear()

        self.ax.plot(timestamps, self.shown_plot)
        self.fig = plt.figure(figsize=(10, 5))
        # Aktualisieren Sie das Canvas
        self.canvas.draw()
        self.canvas.get_tk_widget().configure(width=480, height=250)

    def go_to_next_view(self, epochs, filename, channels, windowsize, overlap, sfreq, raw_data):
        self.pack_forget()  # destroy view 1
        view_2 = Epochs_Details(self.master, epochs=epochs, filename=filename, channels=channels, windowsize=windowsize, overlap=overlap, sfreq=sfreq, raw_data=raw_data)
        view_2.pack(fill='both', expand=True)

    def open_epochs_view(self, overlap, windowsize, data, filename, channels, sfreq, raw_data):
        epochs = create_windows(overlap, windowsize, data)
        self.go_to_next_view(epochs, filename=filename, channels=channels, windowsize=windowsize, overlap=overlap, sfreq=sfreq, raw_data=raw_data)

    def high_low_filter_set(self, event, state):
        if state == 'l':
            self.low_status = event.widget.get(1.0, 'end')
        if state == 'h':
            self.high_status = event.widget.get(1.0, 'end')
        if state == 's':
            self.smoothing_status = event.widget.get(1.0, 'end')

    def apply_filters(self, raw, channels, sfreq, timestamps, filename):
        raw_copy = raw.copy()
        self.raw_filtered, self.smoothed_plot =  eeg_filter(self.ica_status.get(), self.notch_status.get(), self.low_status, self.high_status, filename, self.normalize_status.get(), self.smoothing_status)
        self.change_plot(timestamps, self.smoothed_plot, channels, sfreq)

    def determine_raw(self, raw):
        if self.raw_filtered == 0:
            print("returned raw")
            return raw
        else:
            print("returned filtered raw")
            return self.raw_filtered

    def determine_raw_data(self, raw_data):
        if self.raw_filtered == 0:
            print("returned raw")
            return raw_data
        else:
            print("returned filtered raw")
            return self.smoothed_plot


class Epochs_Details(Frame):
    def __init__(self, parent, epochs, channels, windowsize, overlap, filename, sfreq, raw_data, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Variables
        self.epochs_channel = channels[0]
        timestamps = [i / sfreq for i in range(len(epochs[0][0]))]
        # Erstellen der drei Frames
        frame1 = Frame(self)
        separator1 = ttk.Separator(self, orient='vertical')
        separator1.place(relx=0.5, rely=0, relwidth=0.0014, relheight=1)
        frame2 = Frame(self)
        separator2 = ttk.Separator(self, orient='horizontal')
        separator2.place(relx=0, rely=0.425, relwidth=1, relheight=0.002)
        frame3 = Frame(self)


        # Verwenden der grid-Methode, um die Frames in einem 2x3-Raster anzuordnen
        frame1.grid(row=0, column=0, columnspan=1, rowspan=3, sticky='nsew')
        frame2.grid(row=0, column=1, columnspan=1, rowspan=3,  sticky='nsew')
        frame3.grid(row=1, column=0, columnspan=2, rowspan=4,  sticky='nsew')

        # Verwenden Sie die weight-Option, um die Größe der Frames in der ersten Zeile zu steuern
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=4)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        go_back_button = Button(frame1, text='Back', command=lambda: self.go_back(filename))
        go_back_button.place(relx=0, rely=0)

        # Information Frame
        information_label = Label(frame1, text='Information')
        information_label.pack(side='top')
        name_label = Label(frame1, text="File:    " + filename)
        name_label.place(relx=0, rely=0.1)
        window_label = Label(frame1, text="Window size:    " + windowsize)
        window_label.place(relx=0, rely=0.15)
        overlap_label = Label(frame1, text="Overlap:    " + str(overlap))
        overlap_label.place(relx=0, rely=0.2)
        epoch_label = Label(frame1, text="Number of Epochs per channel:    " + str(len(epochs[0])))
        epoch_label.place(relx=0, rely=0.25)
        epoch_label = Label(frame1, text="Full Epochs:    " + str(len(epochs)*len(epochs[0])))
        epoch_label.place(relx=0, rely=0.3)


        # Epochs Settings Frame
        epoch_settings_label = Label(frame2, text='Epochs')
        epoch_settings_label.pack(side='top')
        epochs_channel = Label(frame2, text='Select Channel:')
        epochs_channel.place(relx=0, rely=0.1)

        # Erstellen Sie ein ttk.Combobox-Widget
        combo_epochs = ttk.Combobox(frame2, values=channels, state="readonly")
        combo_epochs.set(channels[0])
        combo_epochs.place(relx=0, rely=0.15)
        combo_epochs.bind('<<ComboboxSelected>>', lambda event: self.change_epochs_channel(event))

        start_epochs = Label(frame2, text="Start:")
        start_epochs.place(relx=0, rely=0.2)
        self.start_epochs_text = Text(frame2, bg='grey', width=10, height=1)
        self.start_epochs_text.insert('end', '0')
        self.start_epochs_text.config(font=("Arial", 18))
        self.start_epochs_text.place(relx=0.1, rely=0.2)
        self.start_epochs_text.bind('<KeyRelease>', lambda event: self.change_values(event, 's'))

        end_epochs = Label(frame2, text="End:")
        end_epochs.place(relx=0, rely=0.25)
        self.end_epochs_text = Text(frame2, bg='grey', width=10, height=1)
        self.end_epochs_text.insert('end', '9')
        self.end_epochs_text.config(font=("Arial", 18))
        self.end_epochs_text.place(relx=0.1, rely=0.25)
        self.end_epochs_text.bind('<KeyRelease>', lambda event: self.change_values(event, 'e'))

        self.start_epochs_button = Button(frame2, text='Update', command=lambda: self.update_plot(epochs, timestamps, channels, raw_data))
        self.start_epochs_button.place(relx=0, rely=0.35)

        self.all_epochs_button2 = Button(frame2, text='Previous', command=lambda: self.previous_epochs(epochs, timestamps, channels, raw_data))
        self.all_epochs_button2.place(relx=0.5, rely=0.35)

        self.all_epochs_button = Button(frame2, text='Next', command=lambda: self.next_epochs(epochs, timestamps, channels, raw_data))
        self.all_epochs_button.place(relx=0.7, rely=0.35)

        # Erstellen Sie ein Figure-Objekt und eine Axes-Instanz
        self.fig, self.ax = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(10, 5))



        # Plotte den ersten Graphen in den ersten Subplot (Zeile 0, Spalte 0)
        k = 0
        for i in range(2):
            for j in range(5):
                self.ax[i, j].plot(timestamps, epochs[0][k])
                k +=1

        ymin, ymax = raw_data[0].min(), raw_data[0].max()
        for ax in self.ax.flatten():
            ax.set_ylim(ymin, ymax)
        # Erstellen Sie eine FigureCanvasTkAgg-Instanz
        self.start = self.start_epochs_text.get(1.0, 'end')
        self.end = self.end_epochs_text.get(1.0, 'end')

        self.canvas = backend_tkagg.FigureCanvasTkAgg(self.fig, master=frame3)
        self.canvas.get_tk_widget().place(relx=0, rely=0.05)
        self.canvas.get_tk_widget().bind('<Button-1>', lambda event: self.show_plot(self.start, self.end, timestamps, epochs, channels))
        self.canvas.get_tk_widget().configure(width=1000, height=250)

        self.button_download = Button(frame3, text='Download', command=lambda: self.copy_data_to_file(epochs, channels, raw_data))
        self.button_download.place(relx=0.1, rely=0.9)

    def change_values(self, event, v):
        if v == 's':
            if event.widget.get(1.0, 'end').strip():
                self.start = event.widget.get(1.0, 'end')
        if v == 'e':
            if event.widget.get(1.0, 'end').strip():
                self.end = event.widget.get(1.0, 'end')

    def show_plot(self, start, end, timestamps, epochs, channels):
        print('show_plots')

        if int(end) - int(start) <= 9:
            ind = channels.index(self.epochs_channel)
            plt.close('all')
            self.fig, self.ax = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(10, 5))
            values = list(range(int(start), int(end) + 1))
            k = 0
            for i in range(2):
                for j in range(5):
                    if k < len(values):
                        self.ax[i, j].plot(timestamps, epochs[ind][values[k]])
                    k += 1
            plt.show()

    def go_back(self, filename):
        self.pack_forget()  # destroy view 2
        view_1 = Data_Details(self.master, filename)
        view_1.pack(fill='both', expand=True)

    def change_epochs_channel(self, event):
        self.epochs_channel = event.widget.get()


    def update_plot(self, epochs, timestamps, channels, raw_data):
        if int(self.end) - int(self.start) <= 9:
            ind = channels.index(self.epochs_channel)

            for a in self.ax.flat:
                a.clear()

            # Plotte den ersten Graphen in den ersten Subplot (Zeile 0, Spalte 0)
            values = list(range(int(self.start), int(self.end) + 1))

            ymin, ymax = raw_data[ind].min(), raw_data[ind].max()
            k = 0
            plt.plot(timestamps, epochs[ind][0])
            for i in range(2):
                for j in range(5):
                    if k < len(values) and k < len(epochs):
                        print('k: ', k)
                        self.ax[i, j].clear()
                        self.ax[i, j].plot(timestamps, epochs[ind][values[k]])
                        self.ax[i, j].set_ylim(ymin, ymax)
                    else:
                        self.ax[i, j].set_visible(False)
                    k += 1



            """for ax in self.ax.flatten():
                ax.set_ylim(ymin, ymax)"""

            # Aktualisieren Sie das Canvas
            self.canvas.draw()
            self.canvas.get_tk_widget().configure(width=1000, height=250)

    def next_epochs(self, epochs, timestamps, channels, raw_data):
        if int(self.end) + 10 <= len(epochs[0]):
            self.start = str(int(self.start) + 10)
            self.end = str(int(self.end) + 10)

            self.start_epochs_text.delete(1.0, END)
            self.start_epochs_text.insert(END, self.start)

            self.end_epochs_text.delete(1.0, END)
            self.end_epochs_text.insert(END, self.end)
            self.update_plot(epochs, timestamps, channels, raw_data)

    def previous_epochs(self, epochs, timestamps, channels, raw_data):
        if int(self.start) - 10 >= 0:
            self.start = str(int(self.start) -10)
            self.end = str(int(self.end) - 10)

            self.start_epochs_text.delete(1.0, END)
            self.start_epochs_text.insert(END, self.start)

            self.end_epochs_text.delete(1.0, END)
            self.end_epochs_text.insert(END, self.end)
            self.update_plot(epochs, timestamps, channels, raw_data)

    def copy_data_to_file(self, epochs, channels, raw_data):
        data = epochs
        file_name = "epochs.pickle"
        with open("../../files/epochs_files/" + file_name, 'wb') as f:
            pickle.dump(data, f)
        with open("../../files/epochs_files/" + 'rawdata.pickle', 'wb') as f:
            pickle.dump(raw_data, f)



def get_filenames(path_tuple):
    # Initialize empty lists for the filenames and paths
    filenames = []

    # Iterate over the paths
    for path in path_tuple:
        # Use the os.path.basename() function to get the filename
        filename = os.path.basename(path)
        filenames.append(filename)

    return filenames


if __name__ == "__main__":
    app = EEG_GUI()
    app.title('EEG Clustering')

    # Setze die Höhe und Breite des Fensters und verhindere, dass es verändert wird
    app.geometry("960x540")
    app.resizable(False, False)
    app.mainloop()

