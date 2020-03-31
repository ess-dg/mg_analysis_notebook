import os
import numpy as np
import pandas as pd

import mg_cncs.helper_functions as hf

# =============================================================================
# Masks
# =============================================================================

SignatureMask    = 0xC0000000    # 1100 0000 0000 0000 0000 0000 0000 0000
SubSignatureMask = 0x3FE00000    # 0011 1111 1110 0000 0000 0000 0000 0000

ModuleMask       = 0x00FF0000    # 0000 0000 1111 1111 0000 0000 0000 0000
ChannelMask      = 0x001F0000    # 0000 0000 0001 1111 0000 0000 0000 0000
ADCMask          = 0x00003FFF    # 0000 0000 0000 0000 0011 1111 1111 1111
ExTsMask         = 0x0000FFFF    # 0000 0000 0000 0000 1111 1111 1111 1111
TimeStampMask 	 = 0x3FFFFFFF    # 0011 1111 1111 1111 1111 1111 1111 1111
WordCountMask  	 = 0x00000FFF    # 0000 0000 0000 0000 0000 1111 1111 1111


# =============================================================================
# Dictionary
# =============================================================================

Header        	 = 0x40000000    # 0100 0000 0000 0000 0000 0000 0000 0000
Data          	 = 0x00000000    # 0000 0000 0000 0000 0000 0000 0000 0000
EoE           	 = 0xC0000000    # 1100 0000 0000 0000 0000 0000 0000 0000

DataEvent        = 0x04000000    # 0000 0100 0000 0000 0000 0000 0000 0000
DataExTs         = 0x04800000    # 0000 0100 1000 0000 0000 0000 0000 0000


# =============================================================================
# Bit shifts
# =============================================================================

ChannelShift     = 16
ModuleShift      = 16
ExTsShift        = 30

# =============================================================================
#                               EXTRACT CLUSTERS
# =============================================================================

def extract_clusters(folder_path):
    # Declare masks
    TIMESTAMP_MASK = 0x3FFFFFFF
    ADC_MASK = 0x00003FFF
    # Import adc-to-channel dict
    adc_to_ch_dict = hf.get_adc_to_ch_dict()
    wire_di = adc_to_ch_dict['wires']
    grid_di = adc_to_ch_dict['grids']
    # Import data
    file_names = [f for f in os.listdir(folder_path) if f[-4:] == '.bin']
    file_paths = hf.append_folder_and_files(folder_path + '/', file_names)
    data_files = [None]*len(file_paths)
    size = 0
    for i, file_path in enumerate(file_paths):
        data_files[i] = np.fromfile(file_path, dtype=np.dtype('u4'))
        size += (len(data_files[i]) // 10)
    clusters = np.array([np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int),
                         np.zeros([size], dtype=int)])
    # Cluster data (split data into 10 columns - 1 header, 8 data, 1 eoe)
    start = 0
    for i, data_file in enumerate(data_files):
        length = len(data_file)//10
        matrix_T = np.reshape(data_file, (length, 10))
        matrix = np.transpose(matrix_T)
        clusters[0:8, start:(start+length)] = matrix[1:9, :] & ADC_MASK
        clusters[8, start:(start+length)] = matrix[9, :] & TIMESTAMP_MASK
        start += length
    # Store in DataFrame
    clusters_df = pd.DataFrame({'w_adc_m1': clusters[6],
                                'w_adc_m2': clusters[7],
                                'w_ch_adc_m1': clusters[4],
                                'w_ch_adc_m2': clusters[0],
                                'g_adc_m1': clusters[1],
                                'g_adc_m2': clusters[3],
                                'g_ch_adc_m1': clusters[2],
                                'g_ch_adc_m2': clusters[5],
                                'tof': clusters[8],
                                'w_ch_m1': pd.DataFrame({'a': clusters[4]})['a'].map(wire_di).values,
                                'g_ch_m1': pd.DataFrame({'a': clusters[2]})['a'].map(grid_di).values,
                                'g_ch_m2': pd.DataFrame({'a': clusters[5]})['a'].map(grid_di).values})
    return clusters_df
