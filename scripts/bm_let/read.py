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
    # Declare keys in dictionary
    tof_edges = nxs['raw_data_1']['monitor_1']['time_of_flight'][()]
    tof_centers = (tof_edges[:-1] + tof_edges[1:]) / 2
    bm_dict = {key: [] for key in tof_centers}
    for i in np.arange(1, 9, 1):
        monitor = 'monitor_%d' % i
        histogram = nxs['raw_data_1'][monitor]['data'][()][0][0]
        for tof_center, counts in zip(tof_centers, histogram):
            bm_dict[tof_center].append(counts)
    # Initialize DataFrame
    df = pd.DataFrame(bm_dict)
    return df


# =============================================================================
#                                IMPORT MAPPING
# =============================================================================

def get_pixel_to_xyz_mapping(path):
    # Import data
    nxs = h5py.File(path, 'r')
    # Convert from polar to cartesian coordinates
    r = nxs['raw_data_1']['instrument']['detector_1']['distance'][()]
    theta = nxs['raw_data_1']['instrument']['detector_1']['polar_angle'][()] * ((2*np.pi)/360)
    phi = nxs['raw_data_1']['instrument']['detector_1']['azimuthal_angle'][()] * ((2*np.pi)/360)
    # Save in dictionary
    position_dict = {'x': r * np.sin(theta) * np.cos(phi),
                     'y': r * np.sin(theta) * np.sin(phi),
                     'z': r * np.cos(theta),
                     'r': r,
                     'theta': theta * (360/(2*np.pi)),
                     'phi': phi * (360/(2*np.pi))}
    return position_dict
