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


plt.rcParams["figure.figsize"] = (30,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 

def generate_data(window_size, classes, maximumClassCount):
    """
    Generate X,y data from the MIT-BIH Arrhythmia Database in mitbih_database/ directory. 

    Input:
    :param window_size: size of the sequence to be used for classification 
    :param classes: list of classes to select for classification
    :param maximumClassCount: maximum number of samples per class
    
    Output:
    :return: X, y
    X: ECG data, numpy array of shape (n_patients, ecg_window_size)
    y: labels, numpy array of shape (n_patients, )
    """

    # set path
    path = "mitbih_database/"
    window_size = window_size 
    maximumClassCount = maximumClassCount 

    classes = classes
    n_classes = len(classes)
    classCount = [0]*n_classes

    X = list()
    y = list()

    # Read files
    filenames = list(os.walk(path))[0][2]

    # Split and save .csv , .txt 
    records = list()
    annotations = list()
    filenames.sort()

    # segregating filenames and annotations
    for f in filenames:
        filename, file_extension = os.path.splitext(f)
        
        # *.csv; ECG data are the .csv files
        if(file_extension == '.csv'):
            records.append(path + filename + file_extension)

        # *.txt; annotations are the .txt files
        elif(file_extension == '.txt'):
            annotations.append(path + filename + file_extension)

    # Records
    for r in range(len(records)): # 48 records, loop through each patient record
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
                    if classCount[arrhythmia_index] > maximumClassCount: # avoid overfitting
                        pass
                    else:
                        classCount[arrhythmia_index] += 1
                        if(window_size < pos and pos < (len(signals) - window_size)):
                            beat = signals[pos-window_size+1:pos+window_size]
                            X.append(beat)
                            y.append(arrhythmia_index)

    return X, y

    # convert to numpy array

    X = np.array(X)
    y = np.array(y)

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

    X = np.array(X)
    y = np.array(y)

    return X, y

if __name__ == "__main__":

    N_PATIENTS = 48 
    SAMPLING_RATE = 360 # Hz

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

    windowDuration = 0.88 # seconds
    windowWidthSamples = int(windowDuration * SAMPLING_RATE)
    classes = ALL_BEAT_CLASSES.keys()
    
    X, y = generate_data(windowWidthSamples, classes, 10000)
    save_data(X, y)
    #X, y = load_data()

    print(X.shape, y.shape)
    print(X[0,:])


