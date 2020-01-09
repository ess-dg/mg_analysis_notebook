#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" phs.py:
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np


# ============================================================================
#                                   PHS (1D)
# ============================================================================


def phs_1d_plot(clusters, number_bins, label='', norm=1, ylabel='',
                intervals=None, density=False):
    """
    Histograms the ADC-values from wires and grids individually, and overlays
    the results from indiviual events and clustered events. For the clustered
    events, each data point is a summation of the ADC values contained in that
    cluster, i.e. the summed ADC is being histogrammed.

    Args:
        events (DataFrame): Individual events
        clusters (DataFrame): Clustered events
        number_bins (int): Number of bins to use in the histogram

    Returns:
        fig (Figure): Figure containing 1D PHS plot
    """
    # Declare parameters
    titles = ['Wires', 'Grids']
    limits = [[0, 79], [80, 119]]
    ADC_types = ['wADC', 'gADC']
    weights = norm*np.ones(clusters.shape[0])
    bin_centers_vec = []
    hists = []
    for i, (title, limit, ADC_type) in enumerate(zip(titles, limits, ADC_types)):
        plt.subplot(1, 2, i+1)
        # Set figure properties
        plt.title('PHS (1D) - %s' % title)
        plt.xlabel('Collected charge [ADC channels]')
        plt.ylabel('Counts %s' % ylabel)
        plt.yscale('log')
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        if intervals is not None:
            range = intervals[i]
        else:
            range = None
        # Plot
        hist, bins, *_ = plt.hist(clusters[ADC_type], bins=number_bins,
                                  histtype='step', label='Clusters %s' % label,
                                  zorder=5, weights=weights, range=range,
                                  density=density, color='black')
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_centers_vec.append(bin_centers)
        hists.append(hist)
        #plt.legend(loc=1)
    plt.tight_layout()
    return bin_centers_vec, hists


# =============================================================================
#                                   PHS (2D)
# =============================================================================


def PHS_2D_plot(events, bus_start, bus_stop):
    """
    Histograms the ADC-values from each channel individually and summarises it
    in a 2D histogram plot, where the color scale indicates number of counts.
    Each bus is presented in an individual plot.

    Args:
        events (DataFrame): Individual events

    Returns:
        fig (Figure): Figure containing 2D PHS plot
    """
    def PHS_2D_plot_bus(fig, events, sub_title, vmin, vmax):
        plt.xlabel('Channel')
        plt.ylabel('Charge [ADC channels]')
        plt.title(sub_title)
        bins = [120, 120]
        if events.shape[0] > 1:
            plt.hist2d(events.Ch, events.ADC, bins=bins, norm=LogNorm(),
                       range=[[-0.5, 119.5], [0, 4400]], vmin=vmin, vmax=vmax,
                       cmap='jet'
                       )
        plt.colorbar()

    # Prepare figure
    fig = plt.figure()
    number_detectors = (bus_stop - bus_start)//3 + 1
    fig.set_figheight(5*number_detectors)
    if number_detectors == 1:
        width = (17/3) * ((bus_stop - bus_start) + 1)
        rows = ((bus_stop - bus_start) + 1)
    else:
        width = 17
        rows = 3
    fig.set_figwidth(width)
    # Calculate color limits
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    # Iterate through all buses
    for i, bus in enumerate(range(bus_start, bus_stop+1)):
        events_bus = events[events.Bus == bus]
        # Calculate number of grid and wire events in a specific bus
        wire_events = events_bus[events_bus.Ch < 80].shape[0]
        grid_events = events_bus[events_bus.Ch >= 80].shape[0]
        # Plot
        plt.subplot(number_detectors, rows, i+1)
        sub_title = 'Bus: %d, events: %d' % (bus, events_bus.shape[0])
        sub_title += '\nWire events: %d, Grid events: %d' % (wire_events, grid_events)
        PHS_2D_plot_bus(fig, events_bus, sub_title, vmin, vmax)
    plt.tight_layout()
    return fig


# =============================================================================
#                           PHS (Wires vs Grids)
# =============================================================================


def PHS_wires_vs_grids_plot(ce, bus_start, bus_stop):
    """
    Histograms ADC charge from wires vs grids, one for each bus, showing the
    relationship between charge collected by wires and charge collected by
    grids. In the ideal case there should be linear relationship between these
    two quantities.

    Args:
        ce (DataFrame): Clustered events

    Returns:
        fig (Figure): Figure containing nine 2D PHS plots, showing wire charge
                      versus grid charge histogrammed, one plot for each bus.
    """
    def charge_scatter(fig, ce, sub_title, bus, vmin, vmax):
        plt.xlabel('Collected charge wires [ADC channels]')
        plt.ylabel('Collected charge grids [ADC channels]')
        plt.title(sub_title)
        bins = [200, 200]
        ADC_range = [[0, 10000], [0, 10000]]
        plt.hist2d(ce.wADC, ce.gADC, bins=bins, norm=LogNorm(), range=ADC_range,
                   vmin=vmin, vmax=vmax, cmap='jet')
        plt.colorbar()
        return fig

    # Plot data
    fig = plt.figure()
    number_detectors = ((bus_stop + 1) - bus_start)//3
    fig.set_figheight(4*number_detectors)
    fig.set_figwidth(14)
    # Set color limits
    if ce.shape[0] != 0:
        vmin = 1
        vmax = ce.shape[0] // 4500 + 1000
    else:
        vmin = 1
        vmax = 1
    # Plot
    for i, bus in enumerate(range(bus_start, bus_stop+1)):
        events_bus = ce[ce.Bus == bus]
        sub_title = 'Bus %d\n(%d events)' % (bus, events_bus.shape[0])
        plt.subplot(number_detectors, 3, i+1)
        fig = charge_scatter(fig, events_bus, sub_title, bus, vmin, vmax)
    plt.tight_layout()
    return fig
