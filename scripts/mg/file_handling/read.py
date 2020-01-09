#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read.py:
"""

import os
import struct
import shutil
import zipfile
import re
import numpy as np
import pandas as pd

from mg.helper_functions.misc import mkdir_p

# =============================================================================
#                     DICTIONARY FOR BINARY TRANSLATION
# =============================================================================

# MASKS
TYPE_MASK      =   0xC0000000     # 1100 0000 0000 0000 0000 0000 0000 0000
DATA_MASK      =   0xF0000000     # 1111 0000 0000 0000 0000 0000 0000 0000

CHANNEL_MASK   =   0x00FFF000     # 0000 0000 1111 1111 1111 0000 0000 0000
BUS_MASK       =   0x0F000000     # 0000 1111 0000 0000 0000 0000 0000 0000
ADC_MASK       =   0x00000FFF     # 0000 0000 0000 0000 0000 1111 1111 1111
TIMESTAMP_MASK =   0x3FFFFFFF     # 0011 1111 1111 1111 1111 1111 1111 1111
EXTS_MASK      =   0x0000FFFF     # 0000 0000 0000 0000 1111 1111 1111 1111
TRIGGER_MASK   =   0xCF000000     # 1100 1111 0000 0000 0000 0000 0000 0000

# DICTONARY
HEADER         =   0x40000000     # 0100 0000 0000 0000 0000 0000 0000 0000
DATA           =   0x00000000     # 0000 0000 0000 0000 0000 0000 0000 0000
EOE            =   0xC0000000     # 1100 0000 0000 0000 0000 0000 0000 0000

DATA_BUS_START =   0x30000000     # 0011 0000 0000 0000 0000 0000 0000 0000
DATA_EVENT     =   0x10000000     # 0001 0000 0000 0000 0000 0000 0000 0000
DATA_EXTS      =   0x20000000     # 0010 0000 0000 0000 0000 0000 0000 0000

TRIGGER        =   0x41000000     # 0100 0001 0000 0000 0000 0000 0000 0000

# BIT-SHIFTS
CHANNEL_SHIFT   =   12
BUS_SHIFT       =   24
EXTS_SHIFT      =   30


# =============================================================================
#                                UNZIP DATA
# =============================================================================

def unzip_data(zip_source):
    """ Unzips mestyec .zip-files and extracts data-files:

            1. Extracts data in temporary folder
            2. Selects the relevant file and moves it to a temporary location
            3. Removes the temporary fodler where the rest of zipped data is

    Args:
        zip_source (str): Path to '.zip'-file that contains the data

    Returns:
        '.mesytec'-file path (str): Path to the extracted '.mesytec'-file

    """
    dirname = os.path.dirname(__file__)
    zip_temp_folder = os.path.join(dirname, '../zip_temp_folder/')
    mkdir_p(zip_temp_folder)
    file_temp_folder = os.path.join(dirname, '../')
    destination = ''
    with zipfile.ZipFile(zip_source, "r") as zip_ref:
        zip_ref.extractall(zip_temp_folder)
        temp_list = os.listdir(zip_temp_folder)
        source_file = None
        for temp_file in temp_list:
            if temp_file[-8:] == '.mvmelst':
                source_file = temp_file
        source = zip_temp_folder + source_file
        destination = file_temp_folder + source_file
        shutil.move(source, destination)
        shutil.rmtree(zip_temp_folder, ignore_errors=True)
    return destination


# =============================================================================
#                                IMPORT DATA
# =============================================================================

def import_data(file_path, maximum_file_size_in_mb=np.inf):
    """ Imports mestyec data in three steps:

            1. Reads file as binary and saves data in 'content'
            2. Finds the end of the configuration text, i.e. '}\n}\n' followed
               by 0 to n spaces, then saves everything after this to
               'reduced_content'.
            3. Groups data into 'uint'-words of 4 bytes (32 bits) length

    Args:
        file_path (str): Path to '.mesytec'-file that contains the data

    Returns:
        data (tuple): A tuple where each element is a 32 bit mesytec word

    """

    # Get maximum file size in [bytes]
    ONE_MB_IN_BYTES = (1 << 20)
    maximum_file_size_in_bytes = maximum_file_size_in_mb * ONE_MB_IN_BYTES
    # Assign piece size in [bytes]
    piece_size = 1000 * ONE_MB_IN_BYTES
    # Import data
    with open(file_path, mode='rb') as bin_file:
        # Get first piece of data
        content = bin_file.read(piece_size)
        # Skip configuration text
        match = re.search(b'}\n}\n[ ]*', content)
        start = match.end()
        content = content[start:]
        # Split first piece of data into groups of 4 bytes
        data = struct.unpack('I' * (len(content)//4), content)
        # Repeat for the rest of data
        more_data = True
        imported_data = piece_size
        while more_data and imported_data <= maximum_file_size_in_bytes:
            imported_data += piece_size
            piece = bin_file.read(piece_size)
            if not piece:  # Reached end of file
                more_data = False
            else:
                data += struct.unpack('I' * (len(piece)//4), piece)
    # Remove original file
    os.remove(file_path)
    return data


# =============================================================================
#                               EXTRACT CLUSTERS
# =============================================================================

def extract_clusters(data):
    """ Clusters the imported data and stores it in a DataFrame containing
        coicident events (i.e. candidate neutron events).

        Does this in the following fashion for coincident events:
            1. Reads one word at a time
            2. Checks what type of word it is (HEADER, DATA_BUS_START,
               DATA_EVENT, DATA_EXTS or EOE).
            3. When a HEADER is encountered, 'is_open' is set to 'True',
               signifying that a new event has been started. Data is then
               gathered into a single coincident event until a different bus is
               encountered, in which case a new event is started.
            4. When EOE is encountered the event is formed, and timestamp is
               assigned to it and all the created events under the current
               HEADER. This event is placed in the created dictionary.
            5. After the iteration through data is complete, the dictionary
               containing the coincident events is convereted to a DataFrame.

    Args:
        data (tuple): Tuple containing data, one word per element.

    Returns:
        clusters (DataFrame): DataFrame containing one neutron
                              event per row. Each neutron event has
                              information about: "bus", "time",
                              "tof", "wch", "gch", "wadc", "gadc",
                              "wm", "gm" and "cem".

    """
    size = len(data)
    # Initiate dictionary to store clusters
    ce_dict = {'bus': (-1) * np.ones([size], dtype=int),
               'time': (-1) * np.ones([size], dtype=int),
               'trigger_time': (-1) * np.ones([size], dtype=int),
               'tof': (-1) * np.ones([size], dtype=int),
               'wch': (-1) * np.ones([size], dtype=int),
               'gch': (-1) * np.ones([size], dtype=int),
               'wadc': np.zeros([size], dtype=int),
               'gadc': np.zeros([size], dtype=int),
               'wm': np.zeros([size], dtype=int),
               'gm': np.zeros([size], dtype=int),
               'flag': (-1) * np.ones([size], dtype=int)}
    # Declare temporary boolean variables, related to words
    is_open, is_trigger, is_data, is_exts = False, False, False, False
    # Declare temporary variables, related to events
    previous_bus, bus = -1, -1
    max_adc_w, max_adc_g = 0, 0
    different_bus_flag = 0
    # Declare variables that track time and index for events and clusters
    time, trigger_time, ce_index = 0, 0, -1
    # Iterate through data
    for i, word in enumerate(data):
        # Five possibilities: Header, DataBusStart, DataEvent, DataExTs or EoE.
        if (word & TYPE_MASK) == HEADER:
            is_open = True
            is_trigger = (word & TRIGGER_MASK) == TRIGGER
        elif ((word & DATA_MASK) == DATA_BUS_START) & is_open:
            # Extract Bus
            bus = (word & BUS_MASK) >> BUS_SHIFT
            is_data = True
            # Initiate temporary cluster variables and increase cluster index
            previous_bus = bus
            max_adc_w, max_adc_g = 0, 0
            ce_index += 1
            # Save Bus data for cluster
            ce_dict['bus'][ce_index] = bus
        elif ((word & DATA_MASK) == DATA_EVENT) & is_open:
            # Extract Channel and ADC
            channel = ((word & CHANNEL_MASK) >> CHANNEL_SHIFT)
            adc = (word & ADC_MASK)
            bus = (word & BUS_MASK) >> BUS_SHIFT
            if previous_bus != bus: different_bus_flag = 1
            # Wires have channels between 0->79
            if 0 <= channel <= 79:
                # Save cluster data
                ce_dict['wadc'][ce_index] += adc
                ce_dict['wm'][ce_index] += 1
                # Use wire with largest collected charge as hit position
                if adc > max_adc_w: max_adc_w, ce_dict['wch'][ce_index] = adc, channel ^ 1
            # Grids have channels between 80->119
            elif 80 <= channel <= 119:
                # Save cluster data, and check if current channel collected most charge
                ce_dict['gadc'][ce_index] += adc
                ce_dict['gm'][ce_index] += 1
                # Use grid with largest collected charge as hit position
                if adc > max_adc_g: max_adc_g, ce_dict['gch'][ce_index] = adc, channel
            else:
                pass
        elif ((word & DATA_MASK) == DATA_EXTS) & is_open:
            extended_time_stamp = (word & EXTS_MASK) << EXTS_SHIFT
            is_exts = True
        elif ((word & TYPE_MASK) == EOE) & is_open:
            # Extract time_timestamp and add extended timestamp, if ExTs is used
            time_stamp = (word & TIMESTAMP_MASK)
            time = (extended_time_stamp | time_stamp) if is_exts else time_stamp
            # Update Triggertime, if this was a trigger event
            if is_trigger: trigger_time = time
            # Save cluster data
            ce_dict['time'][ce_index] = time
            ce_dict['trigger_time'][ce_index] = trigger_time
            ce_dict['flag'][ce_index] = different_bus_flag
            # Reset temporary variables, related to data in events
            previous_bus, bus = -1, -1
            max_adc_w, max_adc_g = 0, 0
            # Reset temporary boolean variables, related to word-headers
            is_open, is_trigger, is_data = False, False, False

        # Print progress of clustering process
        if i % 1000000 == 1:
            percentage_finished = int(round((i/len(data))*100))
            print('Percentage: %d' % percentage_finished)

    # Remove empty elements in clusters and save in DataFrame for easier analysis
    for key in ce_dict:
        ce_dict[key] = ce_dict[key][0:ce_index]
    ce_df = pd.DataFrame(ce_dict)
    # Extract tof and convert tof and time to seconds
    ce_df['tof'] = (ce_df['time'] - ce_df['trigger_time']).values * 62.5e-9
    ce_df['time'] = ce_df['time'].values * 62.5e-9
    return ce_df


# =============================================================================
#                               EXTRACT EVENTS
# =============================================================================

def extract_events(data):
    size = len(data)
    # Initiate dictionary to store events
    e_dict = {'bus': (-1) * np.ones([size], dtype=int),
              'ch': (-1) * np.ones([size], dtype=int),
              'adc': np.zeros([size], dtype=int)}
    # Declare temporary boolean variables
    is_open, is_data = False, False
    # Declare variable that track index for events
    e_index = 0
    # Iterate through data
    for i, word in enumerate(data):
        # Five possibilities: Header, DataBusStart, DataEvent, DataExTs or EoE.
        if (word & TYPE_MASK) == HEADER:
            is_open = True
        elif ((word & DATA_MASK) == DATA_BUS_START) & is_open:
            is_data = True
        elif ((word & DATA_MASK) == DATA_EVENT) & is_open:
            # Extract Channel and ADC
            channel = ((word & CHANNEL_MASK) >> CHANNEL_SHIFT)
            adc = (word & ADC_MASK)
            bus = (word & BUS_MASK) >> BUS_SHIFT
            # Wires have channels between 0->79
            if 0 <= channel <= 79:
                # Save event data and increase event index and event count
                e_dict['bus'][e_index] = bus
                e_dict['ch'][e_index] = channel ^ 1
                e_dict['adc'][e_index] = adc
                e_index += 1
            # Grids have channels between 80->119
            elif 80 <= channel <= 119:
                # Save event data and increase event index and event count
                e_dict['bus'][e_index] = bus
                e_dict['ch'][e_index] = channel
                e_dict['adc'][e_index] = adc
                e_index += 1
            else:
                pass
        elif ((word & DATA_MASK) == DATA_EXTS) & is_open:
            pass
        elif ((word & TYPE_MASK) == EOE) & is_open:
            # Reset temporary boolean variables, related to word-headers
            is_open, is_trigger, is_data = False, False, False


        # Print progress of clustering process
        if i % 1000000 == 1:
            percentage_finished = int(round((i/len(data))*100))
            print('Percentage: %d' % percentage_finished)

    # Remove empty elements in events and save in DataFrame for easier analysis
    for key in e_dict:
        e_dict[key] = e_dict[key][0:e_index]
    e_df = pd.DataFrame(e_dict)
    return e_df


# =============================================================================
#                                SAVE DATA
# =============================================================================

def save_data(df, path):
    """
    Saves clusters or events to specified HDF5-path.

    Args:
        path (str): Path to HDF5 containing the saved DataFrame
        df (DataFrame): Data

    Yields:
        Data is saved to path.
    """
    # Export to HDF5
    df.to_hdf(path, 'df', complevel=9)


# =============================================================================
#                                LOAD DATA
# =============================================================================

def load_data(path):
    """
    Loads clustered data from specified HDF5-path.

    Args:
        path (str): Path to HDF5 containing the saved data

    Returns:
        df (DataFrame): Data
    """
    df = pd.read_hdf(path, 'df')
    return df


# =============================================================================
#                                FILTER DATA
# =============================================================================

def filter_data(df, parameters):
    """
    Filters clusters based on preferences set on GUI.

    Args:
        ce (DataFrame): Clustered events
        parameters (dict): Dictionary containing information on which
                           parameters to filter on, and within which range.

    Returns:
        ce_red (DataFrame): DataFrame containing the reduced data according to
                            the specifications in "parameters".
    """

    df_red = df
    for parameter, (min_val, max_val, filter_on) in parameters.items():
        if filter_on:
            if parameter == 'layer':
                df_red = df_red[((df_red.wCh % 20) >= min_val) &
                                ((df_red.wCh % 20) <= max_val)]
            elif parameter == 'row':
                df_red = df_red[(((df_red.Bus * 4) + df_red.wCh//20) >= min_val) &
                                (((df_red.Bus * 4) + df_red.wCh//20) <= max_val)]
            else:
                df_red = df_red[(df_red[parameter] >= min_val) &
                                (df_red[parameter] <= max_val)]
    return df_red
