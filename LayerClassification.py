import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
from NNDefs import prepared_flat_file_lines, train_test_split, build_and_train_class_nn
from ROOTDefs import get_po_signal_et_background_files, get_reco_stats

# Define and open path to the flat file that we are reading Et information from
flat_file_path = os.path.join(os.path.expanduser('~'), 'TauTrigger', 'Formatted Data Files', 'Flat Files', 'Classification', 'Classification_PO_Flat.txt')

# Define and open path to the flat file that we will write predicted end values to
pred_file_path = os.path.join(os.path.expanduser('~'), 'TauTrigger', 'Formatted Data Files', 'Flat Files', 'Classification', 'Classification_Tester_Predictions.txt')

tsig, fsig, tback, fback = get_po_signal_et_background_files()

max_et, min_et, avg_et, stdev_et = get_reco_stats(tsig, tback)
print(max_et, min_et, avg_et, stdev_et)
stdev_mult = 10
stdev_div = stdev_et * stdev_mult
shift = -avg_et / stdev_div

cell_ets = []
sig_back = []

all_lines = prepared_flat_file_lines(flat_file_path)

# Convert each line from comma-delimited string to list
all_lines = [line.split(',') for line in all_lines]

# Shuffle signal and background events together
random.shuffle(all_lines)

# Split all_lines into a list for layer ets and a list for signal/background identifier
for line in all_lines:
    floats = [(float(et) / stdev_div) - (float(shift) / stdev_div) for et in line[0:5]]
    cell_ets.append(floats)
    sig_back.append([int(line[5])])

# Split et and identifier lists into training (80%) and test (20%) samples
train_ets, test_ets = train_test_split(cell_ets)
train_sig_back, test_sig_back = train_test_split(sig_back)

# Convert Et lists to numpy arrays for use with tensorflow
cell_ets = np.array(cell_ets)
sig_back = np.array(sig_back)

np.random.seed(6)

model = build_and_train_class_nn(train_ets, train_sig_back, test_ets, test_sig_back, epochs=10)

predicted_values = model.predict(cell_ets)

file_info = \
    '********* \n' \
    '** Signal file: ztt_Output_formatted.root \n' \
    '** Background file: output_MB80_formatted.root \n' \
    '** Input flat file: Classification_PO_Flat.txt \n' \
    '** Network: No hidden layers, no bias \n' \
    '** Epochs: 10 \n' \
    '** Learning Rate: 0.1 \n' \
    '** Value 1: Binary output after 10-normalizing and running through classification network \n' \
    '** Value 2: Binary true signal/background identifier \n' \
    '********* \n'

with open(pred_file_path, 'w') as f:
    f.write(file_info)

    for i, val in enumerate(predicted_values):
        line = str(val[0]) + ',' + str(sig_back[i][0]) + '\n'
        f.write(line)


