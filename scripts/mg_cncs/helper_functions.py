import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_delimiter_table():
    # Import excel files
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../tables/histogram_delimiters.xlsx')
    matrix = pd.read_excel(path).values
    # Save delimiters in dictionary
    wires, grids = [], []
    for row in matrix[1:]:
        if not np.isnan(row[0]):
            wires.append(np.array([row[0], row[1]]))
        if not np.isnan(row[2]):
            grids.append(np.array([row[2], row[3]]))
    delimiters_dict = {'wires': np.array(wires), 'grids': np.array(grids)}
    return delimiters_dict

def import_channel_mappings():
    # Import excel files
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../../tables/grid_wire_channel_mapping.xlsx')
    matrix = pd.read_excel(path).values
    # Save channel mappings in dictionary
    wires, grids = {}, {}
    for row in matrix[1:]:
        if not np.isnan(row[0]):
            wires.update({row[0]: row[1]})
        if not np.isnan(row[2]):
            grids.update({row[2]: row[3]})
    channel_mapping_dict = {'wires': wires, 'grids': grids}
    return channel_mapping_dict


def get_adc_to_ch_dict():
    # Declare parameters
    delimiters_dict = import_delimiter_table()
    channel_mapping_dict = import_channel_mappings()
    intervals_vec = [16, 96]
    # Prepare storage of mapping
    adc_to_ch_dict = {'wires': {i: -1 for i in range(4096)},
                      'grids': {i: -1 for i in range(4096)}}
    # Create main dictionary - ADC->Channel
    for (key, delimiters), intervals in zip(delimiters_dict.items(), intervals_vec):
        for i, (start, stop) in enumerate(delimiters):
            # Get channel mapping and delimiters
            small_delimiters = np.linspace(start, stop, intervals+1)
            # Iterate through small delimiters
            previous_value = small_delimiters[0]
            for j, value in enumerate(small_delimiters[1:]):
                channel = channel_mapping_dict[key][i*intervals+j]
                #print('i: %s, Ch: %s' % (str(i*layers+j), str(channel)))
                start, stop = int(round(previous_value)), int(round(value))
                # Assign ADC->Ch mapping for all values within interval
                for k in np.arange(start, stop, 1):
                    adc_to_ch_dict[key][k] = channel
                previous_value = value
    return adc_to_ch_dict


def append_folder_and_files(folder, files):
    folder_vec = np.array(len(files)*[folder])
    return np.core.defchararray.add(folder_vec, files)


def get_measurement_time(folder_path):
    # Iterate through all files in folder
    file_names = [f for f in os.listdir(folder_path) if f[-4:] == '.bin']
    file_paths = append_folder_and_files(folder_path + '/', file_names)
    duration = len(file_paths) * 60
    return duration
