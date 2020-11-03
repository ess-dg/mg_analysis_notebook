#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read.py: Reads data collected with LET helium-3 tubes
"""

import os
import struct
import shutil
import zipfile
import re
import numpy as np
import pandas as pd
import h5py

# =============================================================================
#                                  IMPORT DATA
# =============================================================================

def import_data(path):
    # Import data
    nxs = h5py.File(path, 'r')
    tof_bin_edges = nxs['mantid_workspace_1']['workspace']['axis1'][()]
    tof_bin_centers = (tof_bin_edges[:-1] + tof_bin_edges[1:]) / 2
    histograms = nxs['mantid_workspace_1']['workspace']['values'][()].transpose()
    # Prepare dictionary
    histogram_dict = {key: value for key, value in zip(tof_bin_centers, histograms)}
    # Initialize DataFrame
    df = pd.DataFrame(histogram_dict)
    return df


# =============================================================================
#                                IMPORT MAPPING
# =============================================================================

def get_pixel_to_xyz_mapping(path):
    # Import data
    nxs = h5py.File(path, 'r')
    detector_indicies = nxs['mantid_workspace_1']['instrument']['detector']['detector_index'][()]
    full_coordinates = nxs['mantid_workspace_1']['instrument']['detector']['detector_positions'][()]
    detector_coordinates = full_coordinates[detector_indicies].transpose()
    # Convert from polar to cartesian coordinates
    r = detector_coordinates[0]
    theta = detector_coordinates[1] * ((2*np.pi)/360)
    phi = detector_coordinates[2] * ((2*np.pi)/360)
    # Save in dictionary
    position_dict = {'x': r * np.sin(theta) * np.cos(phi),
                     'y': r * np.sin(theta) * np.sin(phi),
                     'z': r * np.cos(theta),
                     'r': r,
                     'theta': theta,
                     'phi': phi}
    return position_dict
