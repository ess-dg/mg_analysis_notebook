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
    event_id = nxs['raw_data_1']['instrument']['detector_1']['event_id'][()]
    tof = nxs['raw_data_1']['instrument']['detector_1']['event_time_offset'][()]
    # Prepare dictionary
    id_and_tof_dict = {'pixel_id': event_id, 'tof': tof}
    # Initialize DataFrame
    df = pd.DataFrame(id_and_tof_dict)
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
                     'theta': theta,
                     'phi': phi}
    return position_dict
