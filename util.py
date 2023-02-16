import wfdb

import pandas as pd
import numpy as np

#import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize
#from sklearn.metrics import confusion_matrix
#from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
#import itertools
#import collections
import pickle


'''
# Download the record
signals, fields = wfdb.rdsamp('mit-bih/100', sampto=3000)
annotation = wfdb.rdann('mit-bih/100', 'atr', sampto=3000)

print(signals)
print(annotation.sample)

wfdb.plot.plot_items(signal=signals, ann_samp=[annotation.sample])
'''


plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 

ALL_BEAT_CLASSES = {"N": "Normal beat",
                    "L": "Left bundle branch block beat",
                    "R": "Right bundle branch block beat",
                    "A": "Atrial premature beat",
                    "S": "Premature or ectopic supraventricular beat",
                    "V": "Premature ventricular contraction",
                    "e": "Atrial escape beat", 
                    "n": "Supraventricular escape beat",
                    "E": "Ventricular escape beat",
                    "Q": "Unclassifiable beat"}

def generate_data(window_size, classes, path="mitbih_database/"):
    """
    Generate X,y data for each patient ID from the MIT-BIH Arrhythmia Database in mitbih_database/ directory. 

    Input:
    :param window_size: size of the sequence to be used for classification 
    :param classes: list of classes to select for classification
    :param path: path to the mitbih_database/ directory    

    Output:
    :return: X, y
    X: ECG dictionary, key: patient ID, value: numpy array of shape (n_beats, ecg_window_size)
    y: labels, numpy array of shape (n_patients, )
    """

    # set path
    path = path
    window_size = window_size 

    classes = classes
    n_classes = len(classes)
    classCount = [0]*n_classes

    X = dict() # key: patientID, value: list of patient's beats
    y = dict() # key: patientID, value: list of patient's beat labels

    # Read files
    filenames = list(os.walk(path))[0][2]

    # Split and save .csv , .txt 
    records = list()
    patientIDs = list()
    annotations = list()
    filenames.sort()

    # segregating filenames and annotations
    for f in filenames:
        filename, file_extension = os.path.splitext(f)
        
        # *.csv; ECG data are the .csv files
        if(file_extension == '.csv'):
            records.append(path + filename + file_extension)
            patientIDs.append(int(filename))

        # *.txt; annotations are the .txt files
        elif(file_extension == '.txt'):
            annotations.append(path + filename + file_extension)

    # Records
    for r, id in enumerate(patientIDs): # 48 records, loop through each patient record
    # for r in range(2, 3):
        signals = []

        with open(records[r], 'rt') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file
            row_index = -1
            for row in spamreader:
                if(row_index >= 0): # skip first row
                    MLII_lead = int(row[1]) # Modified Limb Lead (MLII)
                    V5 = int(row[2]) # V5 lead
                    signals.insert(row_index, MLII_lead)
                row_index += 1

        # Read anotations: R-wave position and Arrhythmia class
        with open(annotations[r], 'r') as annotationFile:
            data = annotationFile.readlines() # 650001 lines
            beat = list()

            for d in range(1, len(data)): # skip first row
                splitted = data[d].split(' ')
                splitted = filter(None, splitted)
                next(splitted)                   # first get the sample Time
                pos = int(next(splitted))        # then get the Sample ID
                arrhythmia_type = next(splitted) # lastly get the Arrhythmia type 
                if(arrhythmia_type in classes):
                    arrhythmia_index = classes.index(arrhythmia_type)
                    if(window_size < pos and pos < (len(signals) - window_size)):
                        beat = signals[pos-window_size+1:pos+window_size]
                        if X.get(id) is None:
                            X[id] = list() 
                            y[id] = list()
                        X[id].append(beat)
                        y[id].append(arrhythmia_type)

    return X, y


def plot_patient_beat(X, y, patientID, beat_index):
    """
    Plot patientID's beat

    Input:
    :param X: ECG dictionary, key: patient ID, value: numpy array of shape (n_beats, ecg_window_size)
    :param y: labels, numpy array of shape (n_patients, )
    :param patientID: patient ID
    :param beat_index: beat index
    """

    plt.plot(X[patientID][beat_index])
    plt.title('Patient ID: ' + str(patientID) + ', Beat: ' + str(beat_index) + ', Label: ' + ALL_BEAT_CLASSES[y[patientID][beat_index]], fontsize=20)
    plt.show()

def get_patient_beat(X, y, patientID, beat_index):
    """
    Get patientID's beat

    Input:
    :param X: ECG dictionary, key: patient ID, value: numpy array of shape (n_beats, ecg_window_size)
    :param y: labels, numpy array of shape (n_patients, )
    :param patientID: patient ID
    :param beat_index: beat index

    Output:
    :return: beat, label
    beat: numpy array of shape (1, ecg_window_size)
    label: string
    """

    beat = np.array(X[patientID][beat_index])
    label = y[patientID][beat_index]

    return beat, label

def generate_numpy_from_dict(X_dict, y_dict):
    """
    Generate numpy array from dictionary

    Input:
    :param X_dict: ECG dictionary, key: patient ID, value: numpy array of shape (n_beats, ecg_window_size)
    :param y_dict: key: patient ID, value: labels, numpy array of shape (n_patients, )

    Output:
    :return: X, y
    X: numpy array of shape (n_beats, ecg_window_size)
    y: numpy array of shape (n_beats, )
    """

    X = np.array([])
    y = np.array([])

    for key in X_dict.keys():
        if(X.size == 0):
            X = np.array(X_dict[key])
            y = np.array(y_dict[key])
        else:
            X = np.concatenate((X, np.array(X_dict[key])), axis=0)
            y = np.concatenate((y, np.array(y_dict[key])), axis=0)

    return X, y

def read_arduino_ECG_data(file):
    # generate docstring 
    """
    Read ECG data from .csv file generated from Arduino

    Input:
    :param file: path to .csv file

    Output:
    :return: signal
    signal: numpy array of shape (n_samples, )
    """

    signal = np.array([]) 
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file
        row_index = -1
        for row in reader:
            if(row_index >= 0): # skip first row
                line = row[0].strip('"') 
                lead_ECG = int(line) # Modified Limb Lead (MLII)
                signal.append(lead_ECG)
            row_index += 1

    return signal

def save_data(X, y):
    # save X and y using pickle
    with open('X.pickle', 'wb') as f:
        pickle.dump(X, f)

    with open('y.pickle', 'wb') as f:
        pickle.dump(y, f)

def load_data():
    # load pickle
    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)

    with open('y.pickle', 'rb') as f:
        y = pickle.load(f)

    # convert to numpy array 


    return X, y

if __name__ == "__main__":

    path = "small_test_database/"
    X, y = generate_data(160, ["N"], 1000000, path)
    pass


