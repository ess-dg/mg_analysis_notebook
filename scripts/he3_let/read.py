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
    nxs = h5py.File(path, 'r')
    tof_values = nxs['mantid_workspace_1']['workspace']['axis1'].value
    id_values = nxs['mantid_workspace_1']['workspace']['axis2'].value
    histogram_values = nxs['mantid_workspace_1']['workspace']['values'].value

# =============================================================================
#                                IMPORT MAPPING
# =============================================================================

def get_pixel_to_xyz(path):
    nxs = h5py.File(path, 'r')
    detector_index = nxs['mantid_workspace_1']['instrument']['detector']['detector_index'].value
    detector_positions = nxs['mantid_workspace_1']['instrument']['detector']['detector_positions'].value
    
