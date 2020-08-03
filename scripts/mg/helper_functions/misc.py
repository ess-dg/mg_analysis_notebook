#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misc.py: Helper functions for handling of paths and folders.
"""

from errno import EEXIST
from os import makedirs,path
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



# =============================================================================
#                                 GAUSSIAN FIT
# =============================================================================

def fit_data(hist, bins, a_guess, x0_guess, sigma_guess):
    def Gaussian(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt, pcov = curve_fit(Gaussian, bins, hist, p0=[a_guess, x0_guess, sigma_guess])
    a, x0, sigma = popt[0], popt[1], abs(popt[2])
    perr = np.sqrt(np.diag(pcov))
    length = abs(bins[-1] - bins[0])
    xx = np.linspace(bins[0]-0.5*length, bins[-1]+0.5*length, 1000)
    return a, x0, sigma, xx, Gaussian(xx, a, x0, sigma), perr


# =============================================================================
#                       GET ESTIMATION OF FIT PARAMETERS
# =============================================================================

def get_fit_parameters_guesses(hist, bins):
    # Extract relavant parameters
    maximum = max(hist)
    maximum_idx = find_nearest(hist, maximum)
    half_maximum = maximum/2
    half_maximum_idx_1 = find_nearest(hist[:maximum_idx], half_maximum)
    half_maximum_idx_2 = find_nearest(hist[maximum_idx:], half_maximum) + maximum_idx
    FWHM = bins[half_maximum_idx_2] - bins[half_maximum_idx_1]
    # Calculate guesses
    a_guess = maximum
    x0_guess = bins[maximum_idx]
    sigma_guess = FWHM/(2*np.sqrt(2*np.log(2)))
    return a_guess, x0_guess, sigma_guess


# =============================================================================
#                            CALCULATE PEAK AREA
# =============================================================================

def get_peak_area(energies, x0, sigma, bin_width,
                  peak_lower_limit, peak_upper_limit,
                  background_lower_limit, background_upper_limit):
    """

    """

    # Extract number of counts from regions of interest
    peak_indexes = ((energies >= (x0 + peak_lower_limit*sigma)) &
                    (energies <= (x0 + peak_upper_limit*sigma)))
    background_indexes = ((energies >= (x0 + background_lower_limit*sigma)) &
                         (energies <= (x0 + background_upper_limit*sigma)))
    peak_counts = energies[peak_indexes]
    background_counts = energies[background_indexes]

    # Rename for easier calculation of uncertainties
    a = len(peak_counts)
    b = len(background_counts)
    background_range_in_meV = sigma*abs((background_upper_limit-background_lower_limit))

    # Define normalization constants
    norm = (1/background_range_in_meV) * sigma * (peak_upper_limit-peak_lower_limit)

    # Calculate area by removing constant background level
    c = a - b * norm

    # Calculate uncertainites
    da = np.sqrt(a)
    db = np.sqrt(b)
    dc = np.sqrt(da ** 2 + (db*norm) ** 2)
    area = c
    uncertainty = dc

    # Calculate background to cross-check calculation
    background_level = b*(1/background_range_in_meV)*bin_width

    return area, uncertainty, background_level


# =============================================================================
#                            CALCULATE PEAK AREA - VERSION 2
# =============================================================================

def get_peak_area_2(energies, energies_b, x0, sigma, bin_width,
                    peak_lower_limit, peak_upper_limit,
                    norm, norm_b):
    """

    """

    # Extract number of counts from regions of interest
    peak_indexes = ((energies >= (x0 + peak_lower_limit*sigma)) &
                    (energies <= (x0 + peak_upper_limit*sigma)))
    background_indexes = ((energies_b >= (x0 + peak_lower_limit*sigma)) &
                          (energies_b <= (x0 + peak_upper_limit*sigma)))
    peak_counts = energies[peak_indexes]
    background_counts = energies_b[background_indexes]

    # Rename for easier calculation of uncertainties
    a = len(peak_counts)
    b = len(background_counts)
    background_range_in_meV = sigma*abs((peak_upper_limit-peak_lower_limit))

    # Calculate area by removing constant background level
    c = a * norm - b * norm_b

    # Calculate uncertainites
    da = np.sqrt(a)
    db = np.sqrt(b)
    dc = np.sqrt((da*norm) ** 2 + (db*norm) ** 2)
    area = c
    uncertainty = dc

    # Calculate background to cross-check calculation
    background_level = b*norm_b*(1/background_range_in_meV)*bin_width

    return area, uncertainty, background_level


# =============================================================================
#                            CREATE DIRECTORY
# =============================================================================

def mkdir_p(my_path):
    """
    Creates a directory, equivalent to using mkdir -p on the command line.

    Args:
        my_path (str): Path to where the new folder should be created.

    Yields:
        A new folder at the requested path.
    """
    try:
        makedirs(my_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else: raise


# =============================================================================
#                       FIND NEAREST ELEMENT IN ARRAY
# =============================================================================

def find_nearest(array, value):
    """
    Returns the index of the element in 'array' which is closest to 'value'.

    Args:
        array (numpy array): Numpy array with elements
        value (float): Value which we want to find the closest element to in
                       arrray

    Returns:
        idx (int): index of the element in 'array' which is closest to 'value'
    """
    idx = (np.abs(array - value)).argmin()
    return idx


# =============================================================================
#                          GET MEASUREMENT DURATION
# =============================================================================

def get_duration(df):
    times = df.time.values
    diff = np.diff(times)
    resets = np.where(diff < 0)
    duration_in_TDC_channels = sum(times[resets]) + times[-1]
    duration_in_seconds = duration_in_TDC_channels * 62.5e-9
    return duration_in_seconds


# =============================================================================
#                       APPEND FOLDER AND FILES
# =============================================================================

def append_folder_and_files(folder, files):
    folder_vec = np.array(len(files)*[folder])
    return np.core.defchararray.add(folder_vec, files)


# =============================================================================
#                      GET HISTOGRAM AND BIN CENTERS
# =============================================================================

def get_hist(energies, number_bins, start, stop, weights=None):
    hist, bin_edges = np.histogram(energies, bins=number_bins,
                                   range=[start, stop], weights=weights)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return hist, bin_centers


# =============================================================================
#                             CUSTOMIZE THICK LABELS
# =============================================================================

def set_thick_labels(thickness):
    # Customize matplotlib font sizes
    plt.rc('font', size=thickness)          # controls default text sizes
    plt.rc('axes', titlesize=thickness)     # fontsize of the axes title
    plt.rc('axes', labelsize=thickness)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=thickness)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=thickness)    # fontsize of the tick labels
    plt.rc('legend', fontsize=thickness)    # legend fontsize
    plt.rc('figure', titlesize=thickness)  # fontsize of the figure title

# =============================================================================
#                       ENERGY-WAVELENGTH CONVERSION
# =============================================================================

def meV_to_A(energy):
    return np.sqrt(81.81/energy)


def A_to_meV(wavelength):
    return (81.81/(wavelength ** 2))


# =============================================================================
#                      FIND REDUCED E BASED ON LOCATION
# =============================================================================

def calculate_distance_borders(bins, hist, d=28.413):
    def E_to_v(energy_in_meV):
        # Define constants
        JOULE_TO_meV = 6.24150913e18 * 1000
        meV_TO_JOULE = 1/JOULE_TO_meV
        NEUTRON_MASS = 1.674927351e-27
        # Calculate velocity of neutron
        v = np.sqrt((2*energy_in_meV*meV_TO_JOULE)/NEUTRON_MASS)
        return v

    def get_new_E(d, ToF, ToF_extra):
        # Define constants
        JOULE_TO_meV = 6.24150913e18 * 1000
        NEUTRON_MASS = 1.674927351e-27
        E_new = ((NEUTRON_MASS/2)*(d/(ToF+ToF_extra)) ** 2) * JOULE_TO_meV
        return E_new

    # Declare intervals, in m
    distances = np.arange(0, 31, 5) * 1e-2
    # Extract average E
    average_E = bins[hist == max(hist)]
    average_v = E_to_v(average_E)
    # Calculate additional ToF for these distances
    ToF_extras = distances / average_v
    # Calculate reduced energy from additional ToF (d is from closest voxel)
    ToF = d/average_v
    E_reduced = {}
    for distance in distances:
        E_reduced.update({distance: 0})
    for ToF_extra, distance in zip(ToF_extras, distances):
        E_new = get_new_E(d, ToF, ToF_extra)
        E_reduced[distance] = E_new
    return E_reduced, distances
