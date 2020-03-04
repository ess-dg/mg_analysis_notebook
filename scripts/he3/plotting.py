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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.signal import find_peaks


# =============================================================================
#                                    PHS (1D)
# =============================================================================

def phs_1d_plot(df, number_bins, title):
    fig = plt.figure()
    plt.hist(df['adc'], histtype='step', color='blue', zorder=5,
             bins=number_bins)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Charge (ADC Channels)')
    plt.ylabel('Counts')
    plt.title('PHS - %s' % title)
    return fig

# =============================================================================
#                               CHANNEL HISTOGRAM
# =============================================================================

def He3_Ch_plot(df):
    plt.hist(df['Ch'], histtype='step', color='red', zorder=5, bins=20)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.title('Channel')

# =============================================================================
#                                     TOF
# =============================================================================

def tof_histogram(df, number_bins, label=None, interval=None, color=None):
    # Declare parameters
    time_offset = (0.6e-3) * 1e6
    period_time = (1/14) * 1e6
    hist, bin_edges, *_ = plt.hist((df.tof * 1e6 + time_offset) % period_time,
                                   histtype='step', zorder=5, bins=number_bins,
                                   label=label, range=interval, color=color,
                                   #weights=(1/89670)*np.ones(len(df.tof))
                                   )
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.xlabel('ToF (µs)')
    plt.ylabel('Counts')
    #plt.yscale('log')
    plt.title('ToF')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    #return hist, bin_centers


# =============================================================================
#                                   ENERGY
# =============================================================================


def energy_plot(energy, number_bins, label, start=1, stop=10, useMaxNorm=False,
                color=None, scaling=1):
    """
    Histograms the energy transfer values from a measurement

    Args:
        df (DataFrame): Clustered events
        Ei (float): Incident energy in meV
        number_bins (int): Number of bins to histogram energy transfer data

    Returns:
        fig (Figure): Figure containing nine 2D coincidences histograms, one
                      for each bus.
        dE_hist (numpy array): Numpy array containing the histogram data
        bin_centers (numpy array): Numpy array containing the bin centers
    """
    def meV_to_A(energy):
        return np.sqrt(81.81/energy)

    def A_to_meV(wavelength):
        return (81.81/(wavelength ** 2))

    # Select normalization
    if useMaxNorm is False:
        norm = scaling * np.ones(len(energy))
    else:
        hist_temp, _ = np.histogram(energy, bins=number_bins,
                                    range=[A_to_meV(stop), A_to_meV(start)])
        norm = (1/max(hist_temp))*np.ones(len(energy))
    # Plot data
    plt.xlabel('Energy (meV)')
    plt.title('Energy Distribution')
    plt.xscale('log')
    hist, bin_edges, *_ = plt.hist(energy, bins=number_bins,
                                   range=[A_to_meV(stop), A_to_meV(start)],
                                   zorder=5, histtype='step',
                                   label=label,
                                   weights=norm,
                                   linestyle='-',
                                   color=color)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.ylabel('Counts')
    plt.yscale('log')
    return hist, bin_centers


# =============================================================================
#                                WAVELENGTH
# =============================================================================

def wavelength_plot(energy, number_bins, label, start=1, stop=10,
                    useMaxNorm=False, color=None, scaling=1):
    """
    Histograms the energy transfer values from a measurement

    Args:
        df (DataFrame): Clustered events
        Ei (float): Incident energy in meV
        number_bins (int): Number of bins to histogram energy transfer data

    Returns:
        fig (Figure): Figure containing nine 2D coincidences histograms, one
                      for each bus.
        dE_hist (numpy array): Numpy array containing the histogram data
        bin_centers (numpy array): Numpy array containing the bin centers
    """
    def meV_to_A(energy):
        return np.sqrt(81.81/energy)

    def A_to_meV(wavelength):
        return (81.81/(wavelength ** 2))

    # Select normalization
    if useMaxNorm is False:
        norm = scaling * np.ones(len(energy))
    else:
        hist_temp, _ = np.histogram(meV_to_A(energy), bins=number_bins,
                                    range=[start, stop])
        norm = (1/max(hist_temp))*np.ones(len(energy))
    # Plot data
    plt.xlabel('Wavelength (Å)')
    hist, bin_edges, *_ = plt.hist(meV_to_A(energy), bins=number_bins,
                                   range=[start, stop], zorder=5,
                                   histtype='step',
                                   label=label,
                                   weights=norm,
                                   linestyle='-',
                                   color=color)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.ylabel('Counts')
    return hist, bin_centers


# ==============================================================================
#                             PILEUP - HELIUM-3
# ==============================================================================

def he3_pileup_plot(df):
    def plot_2D_hist(df, title):
        # Declare parameters
        time_offset = (0.6e-3) * 1e6
        period_time = (1/14) * 1e6
        plt.hist2d(df.ADC,
                   (df.ToF * (8e-9) * 1e6 + time_offset) % period_time,
                   range=[[0, 66000], [0, 71429]],
                   bins=[50, 50],
                   norm=LogNorm(),
                   #vmin=vmin, vmax=vmax,
                   cmap='jet')
        plt.title(title)
        plt.ylabel('ToF [µs]')
        plt.xlabel('Charge [ADC channels]')
        cbar = plt.colorbar()
        cbar.set_label('Counts')

    df_full = df
    df_no_pileup = df[df.PileUp == 0]
    df_pileup = df[df.PileUp == 1]
    dfs = [df_no_pileup, df_pileup]
    titles = ['No pile-up', 'Pile-up']
    for i, (df, title) in enumerate(zip(dfs, titles)):
        plt.subplot(2, 2, i+1)
        plot_2D_hist(df, title)
        plt.subplot(2, 2, i+3)
        He3_PHS_plot(df, 2000)
