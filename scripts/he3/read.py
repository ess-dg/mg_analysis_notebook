#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImportHe3.py: Imports He3 data taken using the MCA4 Multichannel Analyzer
"""

import os
import struct
import shutil
import zipfile
import re
import numpy as np
import pandas as pd

# =============================================================================
#                                EXTRACT DATA
# =============================================================================

def extract_events(file_path):
    """ Imports MCA4 data. This is ascii encoded hex and is in 64 bit "words".

    Hex 1->4: Charge amplitude
    Hex 5->15: Time
    Hex 16: Channel and pile up

    Args:
        file_path (str): Path to '.mesytec'-file that contains the data

    Returns: he3_df (DataFrame): DataFrame containing data
    """
    # Masks
    CHANNEL_MASK  = 0x0000000000000003
    PILE_UP_MASK  = 0x000000000000000C
    TIME_MASK     = 0x0000FFFFFFFFFFF0
    ADC_MASK      = 0xFFFF000000000000
    BREAK_MASK    = 0xFFF0000000000000
    # Bit shifts
    CHANNEL_MASK   = 0
    TIME_SHIFT     = 4
    ADC_SHIFT      = 48
    PILE_UP_SHIFT  = 2
    # Import data
    data = np.loadtxt(file_path, dtype='str', delimiter='\n')
    start_idx = np.where(data == '[DATA]')[0][0]
    size = len(data)
    # Declare dictionary to store data
    he3_dict = {'ch':  np.empty([size], dtype=int),
                'tof': np.empty([size], dtype=float),
                'adc': np.empty([size], dtype=int),
                'pile_up': np.empty([size], dtype=int)}
    count = 0
    # Extracts information from data
    for i, row in enumerate(data[start_idx+1:]):
        # Convert ASCII encoded HEX to int (shouldn't it be uint?)
        word = int(row, 16)
        # Check if we should save data
        if (word & BREAK_MASK) != 0:
            # Extract values using masks
            he3_dict['ch'][count] = (word & CHANNEL_MASK)
            he3_dict['tof'][count] = ((word & TIME_MASK) >> TIME_SHIFT) * 8e-9
            he3_dict['adc'][count] = (word & ADC_MASK) >> ADC_SHIFT
            he3_dict['pile_up'][count] = (word & PILE_UP_MASK) >> PILE_UP_SHIFT
            count += 1
        # Print progress of clustering process
        if i % 1000 == 1:
            percentage_finished = int(round((i/len(data))*100))
            print('Percentage: %d' % percentage_finished)
    # Only save the events, cut unused rows
    for key in he3_dict:
        he3_dict[key] = he3_dict[key][0:count]
    he3_df = pd.DataFrame(he3_dict)
    return he3_df

# =============================================================================
#                               SAVE DATA
# =============================================================================

def save_data(df, path):
    """
    Saves clustered data to specified HDF5-path.

    Args:
        path (str): Path to HDF5 containing the saved clusters and events
        df (DataFrame): Events

    Yields:
        Clustered data is saved to specified path.
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
        path (str): Path to HDF5 containing the saved clusters and events

    Returns:
        df (DataFrame): Events

    """
    df = pd.read_hdf(path, 'df')
    return df


# =============================================================================
#                                 FILTER DATA
# =============================================================================

def filter_data(df, parameters):
    """
    Filters clusters based on preferences set on GUI.

    Args:
        ce (DataFrame): Clustered events
        parameters (dict): Dictionary containing information about which
                           parameters to filter on, and within which range.

    Returns:
        ce_red (DataFrame): DataFrame containing the reduced data according to
                            the specifications set on the GUI.
    """

    df_red = df
    for parameter, (min_val, max_val, filter_on) in parameters.items():
        if filter_on:
            df_red = df_red[(df_red[parameter] >= min_val) &
                            (df_red[parameter] <= max_val)]
    return df_red
