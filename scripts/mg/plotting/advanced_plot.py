#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" basic_plotting.py:
"""

import plotly as py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import imageio
import shutil
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


from mg.helper_functions.misc import mkdir_p
from mg.helper_functions.energy_calculation import get_energies, get_distances
from mg.helper_functions.misc import meV_to_A, A_to_meV

import he3.energy_calculation as he3_energy
import mg.helper_functions.energy_calculation as mg_energy
import mg.helper_functions.misc as mg_hf

# =============================================================================
#                               COUNT RATE
# =============================================================================

def calculate_count_rate(ToF_values, measurement_time, number_bins):
    """
        Calculates the count rate. Does this in five steps:

        1. Locates peak index (IMPORANT: The prompt peak must be filtered out before
        the procedure starts, if this is more intense than elastic peak)

        2. Finds edges of the elastic peak corresponding to the FWHM

        3. Calculate duration of peak for all periods

        4. Calculate number of counts within peak for all periods

        5. Calculate count rate according to counts/time

        Args:
            ToF_values (numpy array): Numpy array containing all of the ToF values
            corresponding to the clustered events
            measurement_time (float): Measurement time in seconds

        Returns:
            rate (float): Count rate value expressed in Hz
        """
    # Declare constants, period time is in [µs]
    PERIOD_TIME = (1/14)
    # Histogram data
    ToF_hist, bin_edges = np.histogram(ToF_values, bins=number_bins, range=[0, PERIOD_TIME])
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    # Calculate range in ToF equivalent to FWHM of the peak
    peak_idxs = np.where(ToF_hist == max(ToF_hist))
    peak_idx = peak_idxs[len(peak_idxs)//2][0]
    start_idx = find_nearest(ToF_hist[peak_idx-50:peak_idx], ToF_hist[peak_idx]/2)
    stop_idx = find_nearest(ToF_hist[peak_idx:peak_idx+50], ToF_hist[peak_idx]/2)
    start, stop = bin_centers[peak_idx-50+start_idx], bin_centers[peak_idx+stop_idx]
    # Calculate counts in peak
    ToF_values_peak = ToF_values[(ToF_values >= start) & (ToF_values <= stop)]
    counts_peak = len(ToF_values_peak)
    # Calculate peak duration
    number_of_periods = measurement_time/PERIOD_TIME
    duration_peak_per_period = stop - start
    duration_peak = number_of_periods * duration_peak_per_period
    # Calculate rate
    rate = counts_peak/duration_peak
    # Visualize range
    plt.plot(bin_centers[[peak_idx-50+start_idx, peak_idx+stop_idx]],
             ToF_hist[[peak_idx-50+start_idx, peak_idx+stop_idx]],
             color='red', marker='o', linestyle='', zorder=5)
    plt.plot(bin_centers, ToF_hist, '.-', color='black', zorder=4)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('ToF [s]')
    plt.ylabel('Counts')
    plt.title('Maximum instantaneous count rate: %f Hz' % rate)
    # Print statements for debugging purposes
    print()
    print('**** COUNT RATE ****')
    print('------------------')
    print('Start: %f' % start)
    print('Stop: %f' % stop)
    print('Counts peak: %f' % counts_peak)
    print('Number of periods: %f' % number_of_periods)
    print('Duration of peak per period: %f ' % duration_peak_per_period)
    print('Full duration of peak: %f' % duration_peak)
    print('------------------')
    return rate

# =============================================================================
#                      LAYERS INVESTIGATION - TOF
# =============================================================================

def layers_tof(df, title, number_bins=100):
    # Define parameters for ToF-histogram
    time_offset = (0.6e-3) * 1e6
    period_time = (1/14) * 1e6
    #number_bins = 100
    # Define beam hit position
    GRID_MIN = 87
    GRID_MAX = 89
    ROW = 6
    # Iterate through the first ten layers and compare
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(6)
    #plt.title('Layers - %s' % title)
    df_red = df[(((df.bus * 4) + df.wch//20) == ROW) &
                (df.gch >= GRID_MIN) &
                (df.gch <= GRID_MAX)]
    #plt.hist((df_red.tof * 62.5e-9 * 1e6 + time_offset) % period_time,
    #         bins=number_bins,
    #         #range=[28e3, 28.5e3],
    #         #range=[40500, 40900],
    #         range=[19700, 20100],
    #         zorder=4,
    #         histtype='step',
    #         color='black',
    #         label='All')
    for layer in range(0, 20):
        # Filter data so that we are only using data from a single voxel
        df_red = df[((df.wch % 20) == layer) &
                    (((df.bus * 4) + df.wch//20) == ROW) &
                    (df.gch >= GRID_MIN) &
                    (df.gch <= GRID_MAX)]
        # Plot ToF-histogram
        if layer <= 1:
            label = 'Multi-Grid, layer %d' % (layer + 1)
        elif layer == 2:
            label = '...'
        elif layer == 19:
            label = 'Multi-Grid, layer %d' % (layer + 1)
        else:
            label = None
        plt.hist((df_red.tof * 62.5e-9 * 1e6 + time_offset) % period_time,
                 bins=number_bins,
                 #range=[28150, 28.4e3],
                 #range=[40610, 40800],
                 #range=[38500, 38900],
                 range=[19700, 20050],
                 zorder=4,
                 histtype='step',
                 label=label,
                 #weights=(1/9656)*np.ones(len(df_red.tof))
                 )
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('tof (µs)')
    plt.ylabel('Counts')
    #plt.ylim(0, 300)
    #plt.title('ToF from different layers (gCh = 88, row = 6)')
    plt.legend(loc=1)#, title='Layer')
    return fig


# =============================================================================
#                      LAYERS INVESTIGATION - FWHM
# =============================================================================

def investigate_layers_FWHM(peak, df_mg, df_he3, offset_mg, offset_he3):
    def linear(x, k, m):
        return k*x + m

    def Gaussian(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    # Declare parameters
    origin_voxel = [1, 88, 40]
    number_bins = 50
    GRID_MIN = 87
    GRID_MAX = 89
    ROW = 6
    voxel_to_distance_dict = get_distances(origin_voxel, offset_mg)
    source_chopper_to_fermi_chopper_exit = 27.91845
    he3_tube_position = 28.239 - source_chopper_to_fermi_chopper_exit
    # Define region of the peak we want to study
    left, right = mg_hf.A_to_meV(peak + 0.005), mg_hf.A_to_meV(peak - 0.005)
    # Iterate through all layers, a single voxel from each layer, saving the FWHMs
    fig_1 = plt.figure()
    fig_1.set_figwidth(6)
    fig_1.set_figheight(5)
    #plt.subplot(1, 3, 1)
    FWHMs = []
    distances = []
    errors = []


    for layer in range(0, 20):
        # Filter data so that we are only using data from a single voxel
        df_red = df_mg[((df_mg.wch % 20) == layer) &
                       (((df_mg.bus * 4) + df_mg.wch//20) == ROW) &
                       (df_mg.gch >= GRID_MIN) &
                       (df_mg.gch <= GRID_MAX)]
        # Caluclate energies
        energies = mg_energy.get_energies(df_red, offset_mg)
        hist, bins = mg_hf.get_hist(energies, number_bins, left, right)
        a_guess, x0_guess, sigma_guess = mg_hf.get_fit_parameters_guesses(hist, bins)
        fit_left, fit_right = x0_guess - 2 * sigma_guess, x0_guess + 2 * sigma_guess
        hist_fit, bins_fit = mg_hf.get_hist(energies[(energies >= fit_left) & (energies <= fit_right)],
                                            10, fit_left, fit_right)
        a, x0, sigma, __, __, perr = mg_hf.fit_data(hist_fit, bins_fit, a_guess,
                                                    x0_guess, sigma_guess)
        a_err, x0_err, sigma_err = perr
        # Plot
        if layer <= 9:
            label = '%d' % (layer + 1)
        elif layer == 10:
            label = '...'
        elif layer == 19:
            label = '%d' % (layer + 1)
        else:
            label = None
        plt.errorbar(bins, hist, np.sqrt(hist), marker='.', linestyle='-', #fmt='.-',
                    #capsize=5,
                    zorder=5, label=label)
        xx = np.linspace(fit_left, fit_right, 1000)
        norm = (max(hist)/max(Gaussian(xx, a, x0, sigma)))
        plt.plot(xx, Gaussian(xx, a, x0, sigma)*norm, color='black', linestyle='-', label=None, zorder=50)
        # Extract important parameters
        FWHM = 2 * np.sqrt(2*np.log(2)) * sigma
        FWHM_err = 2 * np.sqrt(2*np.log(2)) * sigma_err
        distance = voxel_to_distance_dict[df_red.bus,
                                          df_red.gch,
                                          df_red.wch][0] - source_chopper_to_fermi_chopper_exit
        FWHMs.append(FWHM)
        errors.append(FWHM_err)
        distances.append(distance)
        plt.xlim(left, right)
    # Stylize plot
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Energy (meV)')
    plt.ylabel('Counts')
    #plt.title('Energy Histogram MG, peak at %.2f Å' % peak)
    plt.legend(loc=1, title='Layer')
    #plt.title('Peak at %.2f Å' % peak)
    #plt.yscale('log')

    fig_2 = plt.figure()
    # Plot He-3
    #plt.subplot(1, 3, 2)
    energies_He3 = he3_energy.get_energies(df_he3, offset_he3)
    hist, bins = mg_hf.get_hist(energies_He3, number_bins, left, right)
    a_guess, x0_guess, sigma_guess = mg_hf.get_fit_parameters_guesses(hist, bins)
    fit_left, fit_right = x0_guess - 2 * sigma_guess, x0_guess + 2 * sigma_guess
    hist_fit, bins_fit = mg_hf.get_hist(energies_He3[(energies_He3 >= fit_left) & (energies_He3 <= fit_right)],
                                        10, fit_left, fit_right)
    a, x0, sigma, __, __, perr = mg_hf.fit_data(hist_fit, bins_fit, a_guess, x0_guess, sigma_guess)
    a_err, x0_err, sigma_err  = perr
    # Plot
    plt.errorbar(bins, hist, np.sqrt(hist), marker='.', fmt='.-', capsize=5, linestyle='-', zorder=5, label='He-3', color='red')
    xx = np.linspace(fit_left, fit_right, 1000)
    norm = (max(hist)/max(Gaussian(xx, a, x0, sigma)))
    plt.plot(xx, Gaussian(xx, a, x0, sigma)*norm, color='black', linestyle='-', label=None, zorder=50)
    # Extract important parameters
    FWHM_He3 = 2 * np.sqrt(2*np.log(2)) * sigma
    FWHM_He3_err = 2 * np.sqrt(2*np.log(2)) * sigma_err
    # Stylize plot
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Energy (meV)')
    plt.ylabel('Counts')
    plt.title('Energy Histogram He-3, peak at %.2f Å' % peak)
    plt.legend()
    plt.xlim(left, right)
    #plt.yscale('log')

    fig_3 = plt.figure()
    fig_3.set_figwidth(6)  # 14
    fig_3.set_figheight(5)  #  5
    # Plot third subplot, containing FWHM for the different layers, as well as
    # linear fit on MG data.
    paras, pcov = np.polyfit(distances, FWHMs, 1, w=1/np.array(errors), cov=True)
    k, m = paras[0], paras[1]
    k_err, m_err = np.sqrt(np.diag(pcov))
    upper_k, upper_m = k - k_err, m + m_err
    lower_k, lower_m = k + k_err, m - m_err
    #plt.subplot(1, 3, 3)
    #plt.title('FWHM vs distance')
    plt.ylabel('FWHM (meV)')
    plt.xlabel('Distance to Fermi-chopper (m)')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.errorbar(distances, FWHMs, errors, marker='.', #capsize=5,
                 zorder=5,
                 label='Multi-Grid detector', color='blue', linestyle='')
    plt.errorbar(he3_tube_position, FWHM_He3, FWHM_He3_err,
                 marker='.', color='red', #capsize=5,
                 zorder=5,
                 label='Helium-3 tube')
    start_lim = 0
    end_lim = 1
    xx = np.linspace(start_lim, end_lim, 100)
    plt.plot(xx, linear(xx, upper_k, upper_m), color='blue',
             linestyle='dashed', label=None)
    plt.plot(xx, linear(xx, k, m), color='blue',
             linestyle='dashed', label='Multi-Grid detector, linear fit')
    plt.plot(xx, linear(xx, lower_k, lower_m), color='blue',
             linestyle='dashed', label=None)
    plt.xlim(start_lim,
             end_lim)

    plt.tight_layout()
    #plt.ylim(linear(xx[mg_hf.find_nearest(xx, start_lim)], k, m)*0.9,
    #         linear(xx[mg_hf.find_nearest(xx, end_lim)], k, m)*1.1)
    plt.ylim(0, linear(xx[mg_hf.find_nearest(xx, end_lim)], k, m)*1.1)
    # Get values at He-3 tube
    idx = mg_hf.find_nearest(xx, he3_tube_position)
    interpolated_mg_value_at_he3 = linear(xx[idx], k, m)
    inter_mg_upper = linear(xx[idx], upper_k, upper_m)
    inter_mg_lower = linear(xx[idx], lower_k, lower_m)
    he3_value = FWHM_He3

    # Interpolate He-3 to MG
    mg_first_voxel_position = distances[0]
    FWHM_He3_interpolated = FWHM_He3 * (mg_first_voxel_position/he3_tube_position)
    x_he3_inter = [he3_tube_position, mg_first_voxel_position]
    y_he3_inter = [FWHM_He3, FWHM_He3_interpolated]
    err_he3_iter = [FWHM_He3_err,
                    FWHM_He3_err*(mg_first_voxel_position/he3_tube_position)]
    #plt.errorbar(x_he3_inter, y_he3_inter, err_he3_iter,
    #             marker='s', linestyle='', color='red',
    #             label='FWHM interpolation: Helium-3 tube')
    paras = np.polyfit(x_he3_inter, y_he3_inter, 1,
                       w=1/np.array(err_he3_iter), cov=False)
    k_he3, m_he3 = paras[0], paras[1]
    #plt.plot(xx, linear(xx, k_he3, m_he3), color='red', linestyle='dashed',
    #         label='Fit: Helium-3 tube')
    #plt.title('Peak at %.2f Å' % peak)
    plt.legend(title='FWHM')
    return [fig_1, fig_3], interpolated_mg_value_at_he3, he3_value, k, k_he3, [inter_mg_lower, inter_mg_upper], FWHM_He3_err


# =============================================================================
#                    LAYERS INVESTIGATION - DELTA TOF
# =============================================================================


def investigate_layers_delta_ToF(df_MG, df_He3, origin_voxel):
    def linear(x, k, m):
        return k*x + m

    def Gaussian(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    # Declare parameters
    time_offset = (0.6e-3) * 1e6
    period_time = (1/14) * 1e6
    start = 16.5e3  # us
    stop = 17e3  # us
    number_bins = 500
    GRID = 88
    ROW = 6
    distance_He3 = 28.239
    voxel_to_distance_dict = get_distances(origin_voxel, 0)
    # Get ToF He-3
    ToF_He3 = (df_He3.tof * 1e6 + time_offset) % period_time
    # Declare storage vectors
    FWHMs = []
    distances = []
    errors = []
    # Iterate through all layers
    fig = plt.figure()
    for layer in range(0, 20):
        # Filter data so that we are only using data from a single voxel
        MG_red = df_MG[((df_MG.wCh % 20) == layer) &
                    (((df_MG.Bus * 4) + df_MG.wCh//20) == ROW) &
                    (df_MG.gCh == GRID)]
        ToF_MG = (MG_red.ToF * (62.5e-9) * 1e6 + time_offset) % period_time
        hist_MG, bins_MG = np.histogram(ToF_MG, range=[start, stop], bins=number_bins)
        bin_centers_MG = 0.5 * (bins_MG[1:] + bins_MG[:-1])
        plt.errorbar(bin_centers_MG, hist_MG, np.sqrt(hist_MG), fmt='.-',
                     capsize=5, zorder=20, label='MG', color='blue')
        # Get background level
        background_MG_events = hist_MG[(bin_centers_MG >= 16.5e3) & (bin_centers_MG <= 16.6e3)]
        background_MG_per_bin = sum(background_MG_events)/len(background_MG_events)
        #plt.axhline(y=background_MG_per_bin, color='green', linewidth=2, label=None, linestyle='--')
        # Get FWHM and plot it
        FWHM, idx1, idx2, max_idx = calculate_FWHM(bin_centers_MG, hist_MG)
        plt.plot(bin_centers_MG[max_idx], hist_MG[max_idx], 'o', color='red')
        plt.plot(bin_centers_MG[[idx1, idx2]],
                 [hist_MG[max_idx]/2, hist_MG[max_idx]/2], color='red')
        # Use fitting procedure to get parameters instead
        middle = bin_centers_MG[max_idx]
        #hist_fit, bin_edges_fit = np.histogram(ToF_MG, range=[middle-15, middle+15], bins=50)
        #bins_fit =  0.5 * (bin_edges_fit[1:] + bin_edges_fit[:-1])
        a_guess, x0_guess, sigma_guess = get_fit_parameters_guesses(hist_MG, bin_centers_MG)
        fit_left, fit_right = x0_guess - sigma_guess*2, x0_guess + sigma_guess*2
        hist_fit, bins_fit = get_hist(ToF_MG[(ToF_MG >= fit_left) & (ToF_MG <= fit_right)],
                                      10, fit_left, fit_right)
        a, x0, sigma, __, __, pcov = fit_data(hist_fit, bins_fit, a_guess, x0_guess, sigma_guess)
        a_err, x0_err, sigma_err = np.sqrt(np.diag(pcov))
        FWHM_fit = 2 * np.sqrt(2*np.log(2)) * sigma
        FWHM_err = 2 * np.sqrt(2*np.log(2)) * sigma_err
        xx = np.linspace(fit_left, fit_right, 1000)
        norm = hist_MG[max_idx]/max(Gaussian(xx, a, x0, sigma))
        plt.plot(xx, Gaussian(xx, a, x0, sigma)*norm, color='black',
                 linestyle='-', label=None, zorder=50)
        # Store important values
        distance = voxel_to_distance_dict[MG_red.Bus, MG_red.gCh, MG_red.wCh][0]
        print('Layer: %d, distance: %')
        distances.append(distance)
        FWHMs.append(FWHM_fit)
        errors.append(FWHM_err)
        print('MG velocity [cm/µs], layer %d: %f' % (layer, (distance*100)/bin_centers_MG[max_idx]))
        # Plot He-3
        hist_He3, bins_He3 = np.histogram(ToF_He3, range=[start, stop], bins=number_bins)
        bin_centers_He3 = 0.5 * (bins_He3[1:] + bins_He3[:-1])
        # Calculate FWHM
        FWHM_He3, idx1, idx2, max_idx = calculate_FWHM(bin_centers_He3, hist_He3)
        # Fit data for He-3 instead
        middle = bin_centers_He3[max_idx]
        #hist_fit, bin_edges_fit = np.histogram(ToF_He3, range=[middle-15, middle+15], bins=50)
        #bins_fit =  0.5 * (bin_edges_fit[1:] + bin_edges_fit[:-1])
        a_guess, x0_guess, sigma_guess = get_fit_parameters_guesses(hist_He3, bin_centers_He3)
        fit_left, fit_right = x0_guess - sigma_guess*2, x0_guess + sigma_guess*2
        hist_fit, bins_fit = get_hist(ToF_He3[(ToF_He3 >= fit_left) & (ToF_He3 <= fit_right)],
                                      10, fit_left, fit_right)
        a, x0, sigma, __, __, pcov = fit_data(hist_fit, bins_fit, a_guess, x0_guess, sigma_guess)
        a_err, x0_err, sigma_err = np.sqrt(np.diag(pcov))
        FWHM_he3_fit = 2 * np.sqrt(2*np.log(2)) * sigma
        FWHM_he3_fit_err = 2 * np.sqrt(2*np.log(2)) * sigma_err
        # Plot Gaussian
        xx = np.linspace(fit_left, fit_right, 1000)
        norm = hist_He3[max_idx]/max(Gaussian(xx, a, x0, sigma))
        plt.plot(xx, Gaussian(xx, a, x0, sigma)*norm, color='black', linestyle='-', label=None, zorder=50)
        # Plot data
        plt.errorbar(bin_centers_He3, hist_He3, np.sqrt(hist_He3), fmt='.-',
                     capsize=5, zorder=5, label='He-3', color='red')
        background_He3_events = hist_He3[(bin_centers_He3 >= 16.8e3) & (bin_centers_He3 <= 16.9e3)]
        background_He3_per_bin = sum(background_He3_events)/len(background_He3_events)
        #plt.axhline(y=b, color='red', linewidth=2, label=None, linestyle='--')
        print('Distance in cm: %f' % (distance_He3*100))
        print('Flight time in us: %f' % bin_centers_He3[max_idx])
        print((distance_He3*100)/bin_centers_He3[max_idx])
        plt.legend(loc=2)
        plt.ylim(1, 1e5)
        plt.xlabel('ToF [µs]')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.title('Comparison ∆ToF, layer: %d' % layer)
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        # Save plot
        file_name = '%d.png' % layer
        output_path = output_folder + file_name
        fig.savefig(output_path, bbox_inches='tight')
        plt.close()

    # Plot FWHM
    fig = plt.figure()
    plt.xlabel('Distance [m]')
    plt.ylabel('FWHM [µs]')
    plt.title('FWHM ∆ToF in peaks as a function of distance')
    plt.errorbar(distances, FWHMs, errors, label='MG', fmt='.-', capsize=5, linestyle='', color='blue', zorder=5)
    plt.errorbar(distance_He3, FWHM_he3_fit, FWHM_he3_fit_err, label='He-3',
                 fmt='.-', capsize=5, color='red', linestyle='', zorder=5)

    # Fit MG data
    paras, pcov = np.polyfit(distances, FWHMs, 1, w=1/np.array(errors), cov=True)
    k, m = paras[0], paras[1]
    k_err, m_err = np.sqrt(np.diag(pcov))
    upper_k, upper_m = k - k_err, m + m_err
    lower_k, lower_m = k + k_err, m - m_err
    xx = np.linspace(28, 30, 100)
    plt.plot(xx, linear(xx, upper_k, upper_m), color='black', linestyle='-.', label='Multi-Grid Fit (Upper Bound)')
    plt.plot(xx, linear(xx, k, m), color='black', linestyle='-', label='Multi-Grid Fit')
    plt.plot(xx, linear(xx, lower_k, lower_m), color='black', linestyle='-.', label='Multi-Grid Fit (Lower Bound)')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.legend(loc=2)
    print('ToF spread: %f +/- %f us/m' % (k, k_err))
    fig.show()


# =============================================================================
#                      LAYERS INVESTIGATION - COUNTS
# =============================================================================

def layers_counts(df, duration, title):
    # Get count as a function of layer and grid
    layers = np.arange(0, 20, 1)
    grids = np.arange(80, 120, 1)
    counts_layers = [df[(df.wch % 20) == layer].shape[0] for layer in layers]
    counts_grids = [df[df.gch == grid].shape[0] for grid in grids]
    # Plot data
    fig = plt.figure()
    #fig.suptitle('Counts vs layers - %s' % title)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    plt.subplot(1, 2, 1)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Layer')
    plt.ylabel('Rate (events/s)')
    plt.title('Counts vs layer')
    plt.errorbar(layers, counts_layers/duration, np.sqrt(counts_layers)*(1/duration),
                 fmt='.-', capsize=5, zorder=5, color='black')
    # Look at other direction
    plt.subplot(1, 2, 2)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Grid (bottom to top)')
    plt.ylabel('Rate (events/s)')
    plt.title('Counts vs grid')
    plt.errorbar(grids, counts_grids/duration, np.sqrt(counts_grids)*(1/duration),
                 fmt='.-', capsize=5, zorder=5, color='black')
    plt.tight_layout()
    return fig

# =============================================================================
#                      LAYERS INVESTIGATION - GRIDS
# =============================================================================

def investigate_grids(df, duration):
    # Declare parameters
    time_offset = (0.6e-3) * 1e6
    period_time = (1/14) * 1e6
    # Plot
    fig = plt.figure()
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    gChs = np.arange(80, 120, 1)
    thresholds = [200, 400, 600, 800, 1000, 1200]
    for threshold in thresholds:
        df_red = df[(df.gADC >= threshold) & (df.wADC >= threshold)]
        counts = [df_red[df_red.gCh == gCh].shape[0] for gCh in gChs]
        plt.errorbar(gChs, counts, np.sqrt(counts), zorder=5,
                     label='Threshold: %d ADC channels' % threshold)
    plt.legend()
    plt.xlabel('Grid')
    plt.ylabel('Counts')
    fig.show()
    fig2 = plt.figure()
    plt.hist2d((df.ToF * 62.5e-9 * 1e6 + time_offset) % period_time, df.gCh,
               bins=[500, 40],
               #range=[[0, 7e4], [0.5, 39.5]],
               vmin=8e2, vmax=5e5,
               norm=LogNorm(),
               cmap='jet')
    plt.xlabel('ToF (µs)')
    plt.ylabel('Grid')
    plt.colorbar()
    fig2.show()
    fig3 = plt.figure()
    plt.hist2d(df.gADC, df.gCh,
               bins=[500, 40],
               #range=[[0, 7e4], [0.5, 39.5]],
               #norm=LogNorm(),
               cmap='jet')
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Grid')
    plt.colorbar()
    fig3.show()


# =============================================================================
#                                  ENERGY
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

    # Select normalization
    if useMaxNorm is False:
        norm = scaling * np.ones(len(energy))
    else:
        hist_temp, _ = np.histogram(energy, bins=number_bins,
                                    range=[A_to_meV(stop), A_to_meV(start)])
        norm = (1/max(hist_temp))*np.ones(len(energy))
    # Plot data
    plt.xlabel('Energy (meV)')
    plt.xscale('log')
    hist, bin_edges, *_ = plt.hist(energy, bins=number_bins,
                                   range=[A_to_meV(stop), A_to_meV(start)],
                                   zorder=5, histtype='step',
                                   label=label, weights=norm, color=color)

    #heights_MG_non_coated = [8000, 1000]

    #peaks, heights = get_peaks(hist, heights_MG_non_coated, number_bins)
    #bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    #plt.plot(bin_centers[peaks], hist[peaks], 'x', color='red')
    #plt.plot(bin_centers, heights, color='black')
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.ylabel('Counts')
    return hist, bin_centers


# =============================================================================
#                                  WAVELENGTH
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
                                   color=color)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.ylabel('Normalized counts')
    return hist, bin_centers


# =============================================================================
#                               HELPER FUNCTIONS
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


def calculate_FWHM(bins, hist):
    # Extract relavant parameters
    maximum = max(hist)
    maximum_idx = find_nearest(hist, maximum)
    half_maximum = maximum/2
    half_maximum_idx_1 = find_nearest(hist[:maximum_idx], half_maximum)
    half_maximum_idx_2 = find_nearest(hist[maximum_idx:], half_maximum) + maximum_idx
    FWHM = bins[half_maximum_idx_2] - bins[half_maximum_idx_1]
    return FWHM, half_maximum_idx_1, half_maximum_idx_2, maximum_idx
