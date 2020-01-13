#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" basic_plotting.py:
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio

from mg.helper_functions.mapping import create_mapping
from mg.helper_functions.border_lines import initiate_detector_border_lines
from mg.helper_functions.area_and_solid_angle import get_multi_grid_area_and_solid_angle

# ============================================================================
#                                   PHS (1D)
# ============================================================================


def phs_1d_plot(clusters, number_bins, interval, title):
    """
    Histograms the ADC-values from wires and grids individually, and overlays
    the results from indiviual events and clustered events. For the clustered
    events, each data point is a summation of the ADC values contained in that
    cluster, i.e. the summed ADC is being histogrammed.

    Args:
        clusters (DataFrame): Clustered events
        number_bins (int): Number of bins to use in the histogram

    Yields:
        Figure containing a 1D PHS plot
    """
    # Declare parameters
    fig = plt.figure()
    labels = ['Wires', 'Grids']
    limits = [[0, 79], [80, 119]]
    adc_types = ['wadc', 'gadc']
    # Produce plot
    for i, (label, adc_type) in enumerate(zip(labels, adc_types)):
        # Set figure properties
        plt.title('PHS - %s' % title)
        plt.xlabel('Collected charge [ADC channels]')
        plt.ylabel('Counts')
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        # Plot
        plt.hist(clusters[adc_type], bins=number_bins, histtype='step',
                 zorder=5, range=interval, label=label)
    plt.legend()
    return fig


# =============================================================================
#                                   PHS (2D)
# =============================================================================


def phs_2d_plot(events, bus_start, bus_stop, title):
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
            plt.hist2d(events.ch, events.adc, bins=bins, norm=LogNorm(),
                       range=[[-0.5, 119.5], [0, 4400]], vmin=vmin, vmax=vmax,
                       cmap='jet')
        plt.colorbar()

    # Prepare figure
    fig = plt.figure()
    fig.suptitle('PHS (2D) - %s' % title)
    number_detectors = (bus_stop - bus_start)//3 + 1
    fig.set_figheight(4*number_detectors)
    if number_detectors == 1:
        width = (14/3) * ((bus_stop - bus_start) + 1)
        rows = ((bus_stop - bus_start) + 1)
    else:
        width = 14
        rows = 3
    fig.set_figwidth(width)
    # Calculate color limits
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    # Iterate through all buses
    for i, bus in enumerate(range(bus_start, bus_stop+1)):
        events_bus = events[events.bus == bus]
        # Calculate number of grid and wire events in a specific bus
        wire_events = events_bus[events_bus.ch < 80].shape[0]
        grid_events = events_bus[events_bus.ch >= 80].shape[0]
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


def phs_wires_vs_grids_plot(ce, bus_start, bus_stop, title):
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
        plt.xlabel('Collected charge wires (ADC channels)')
        plt.ylabel('Collected charge grids (ADC channels)')
        plt.title(sub_title)
        bins = [200, 200]
        ADC_range = [[0, 10000], [0, 10000]]
        plt.hist2d(ce.wadc, ce.gadc, bins=bins, norm=LogNorm(), range=ADC_range,
                   vmin=vmin, vmax=vmax, cmap='jet')
        plt.colorbar()
        return fig

    # Plot data
    fig = plt.figure()
    fig.suptitle('PHS (2D) - %s' % title)
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
        events_bus = ce[ce.bus == bus]
        sub_title = 'Bus %d\n(%d events)' % (bus, events_bus.shape[0])
        plt.subplot(number_detectors, 3, i+1)
        fig = charge_scatter(fig, events_bus, sub_title, bus, vmin, vmax)
    plt.tight_layout()
    return fig


# =============================================================================
#                                  ToF
# =============================================================================


def tof_histogram(df, number_bins, title, label=None, norm=1, interval=None):
    """
    Histograms the ToF values in the clustered data.

    Args:
        df (DataFrame): Clustered events
        number_bins (int): The number of bins to histogram the data into

    Yields:
        Figure containing the ToF histogram, this can then be used in overlay
        or in sub plots.
    """
    # Declare parameters
    time_offset = (0.6e-3) * 1e6
    period_time = (1/14) * 1e6
    weights = np.ones(df.shape[0])*norm
    # Prepare figure
    plt.title('ToF - %s' % title)
    plt.xlabel('ToF ($\mu$s)')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Histogram data
    hist, bin_edges, *_ = plt.hist((df.tof * 62.5e-9 * 1e6 + time_offset) % period_time,
                                   bins=number_bins, zorder=4, histtype='step',
                                   label=label, weights=weights, range=interval)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    #return hist, bin_centers

# =============================================================================
#                          Coincidence Histogram (2D)
# =============================================================================

def ce_2d_plot(ce, measurement_time, bus_start, bus_stop, title):
    """
    Histograms the hitposition, expressed in wire- and grid channel coincidences
    in all of the detector voxels.

    Args:
        ce (DataFrame): Clustered events
        measurement_time (float): Measurement time expressed in seconds

    Returns:
        fig (Figure): Figure containing nine 2D coincidences histograms, one
                      for each bus.
        histograms (numpy array): 2D numpy array containing a 2D matrix.
                                  Columns are grids, rows are wires, and the
                                  values in the cells are the number of counts
                                  in that specific voxel.
    """

    def plot_2D_bus(fig, sub_title, ce, vmin, vmax, duration):
        h, *_ = plt.hist2d(ce.wch, ce.gch, bins=[80, 40],
                           range=[[-0.5, 79.5], [79.5, 119.5]],
                           vmin=vmin, vmax=vmax,
                           norm=LogNorm(), cmap='jet')
        plt.xlabel('Wire (Channel number)')
        plt.ylabel('Grid (Channel number)')
        plt.title(sub_title)
        plt.colorbar()
        return fig, h

    # Perform initial filter
    ce = ce[(ce.wch != -1) & (ce.gch != -1)]
    # Calculate color limits
    if ce.shape[0] != 0:
        duration = measurement_time
        vmin = 1
        vmax = ce.shape[0] // 450 + 5
    else:
        duration = 1
        vmin = 1
        vmax = 1
    # Plot data
    fig = plt.figure()
    fig.suptitle('Coincident events - %s' % title)
    number_detectors = (bus_stop - bus_start)//3 + 1
    fig.set_figheight(5*number_detectors)
    if number_detectors == 1:
        width = (14/3) * ((bus_stop - bus_start) + 1)
        rows = ((bus_stop - bus_start) + 1)
    else:
        width = 14
        rows = 3
    fig.set_figwidth(width)
    histograms = []
    for i, bus in enumerate(range(bus_start, bus_stop+1)):
        ce_bus = ce[ce.bus == bus]
        # Calculate number of events and rate in a specific bus
        number_events = ce_bus.shape[0]
        events_per_s = number_events/duration
        events_per_m_s = events_per_s
        sub_title = ('Bus %d\n(%d events, %.3f events/s)' % (bus, number_events, events_per_s))
        plt.subplot(number_detectors, rows, i+1)
        fig, h = plot_2D_bus(fig, sub_title, ce_bus, vmin, vmax, duration)
        histograms.append(h)
    plt.tight_layout()
    return fig, histograms


# =============================================================================
#                           Coincidence Histogram (3D)
# =============================================================================

def ce_3d_plot(df, title):
    """
    Produces a 3D hit-position histogram in (x, y, z)-coordinates, where the
    colorbar indicates number of counts at that specific coordinate.

    Args:
        df (DataFrame): Clustered events

    Yields:
        HTML file containing the 3D-histogram plot, automatically opened in the
        default browser.
    """
    origin_voxel = [1, 88, 40]
    # Declare max and min count
    min_count = 0
    max_count = np.inf
    # Perform initial filters
    df = df[(df.wch != -1) & (df.gch != -1)]
    # Initiate 'voxel_id -> (x, y, z)'-mapping
    mapping = create_mapping(origin_voxel)
    # Initiate border lines
    b_traces = initiate_detector_border_lines(mapping)
    # Calculate 3D histogram
    H, edges = np.histogramdd(df[['wch', 'gch', 'bus']].values,
                              bins=(80, 40, 3),
                              range=((0, 80), (80, 120), (0, 3))
                              )
    # Insert results into an array
    hist = [[], [], [], []]
    loc = 0
    labels = []
    for wCh in range(0, 80):
        for gCh in range(80, 120):
            for bus in range(0, 3):
                over_min = H[wCh, gCh-80, bus] > min_count
                under_max = H[wCh, gCh-80, bus] <= max_count
                if over_min and under_max:
                    coord = mapping[bus, gCh, wCh]
                    hist[0].append(coord['x'])
                    hist[1].append(coord['y'])
                    hist[2].append(coord['z'])
                    hist[3].append(H[wCh, gCh-80, bus])
                    loc += 1
                    distance = np.sqrt(coord['x'] ** 2 + coord['y'] ** 2 + coord['z'] ** 2)
                    labels.append('Module: %d<br>' % bus
                                  + 'WireChannel: %d<br>' % wCh
                                  + 'GridChannel: %d<br>' % gCh
                                  + 'Counts: %d<br>' % H[wCh, gCh-80, bus]
                                  + 'Distance: %.3f' % distance)
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[0],
                               y=hist[1],
                               z=hist[2],
                               mode='markers',
                               marker=dict(size=5,
                                           color=np.log10(hist[3]),
                                           colorscale='Jet',
                                           opacity=1,
                                           colorbar=dict(thickness=20,
                                                         title='log10(counts)'
                                                         ),
                                           #cmin=1,
                                           #cmax=5.5
                                           ),
                               text=labels,
                               name='Multi-Grid',
                               scene='scene1'
                               )
    # Introduce figure and put everything together
    fig = py.tools.make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])
    # Insert histogram
    fig.append_trace(MG_3D_trace, 1, 1)
    # Insert vessel borders
    for b_trace in b_traces:
        fig.append_trace(b_trace, 1, 1)
    # Assign layout with axis labels, title and camera angle
    a = 2
    camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0),
                  eye=dict(x=1*a, y=1*a, z=-1*a))
    scene=dict(camera=camera, #the default values are 1.25, 1.25, 1.25
           xaxis=dict(),
           yaxis=dict(),
           zaxis=dict(),
           aspectmode='data')
    fig['layout']['scene1']['xaxis'].update(title='x [m]')
    fig['layout']['scene1']['yaxis'].update(title='y [m]')
    fig['layout']['scene1']['zaxis'].update(title='z [m]')
    #fig['layout']['scene1']['xaxis'].update(showticklabels=False)
    #fig['layout']['scene1']['yaxis'].update(showticklabels=False)
    #fig['layout']['scene1']['zaxis'].update(showticklabels=False)
    fig['layout'].update(title='Coincidences (3D) - %s' % title)
    fig['layout']['scene1'].update(scene)
    fig.layout.showlegend = False
    py.offline.init_notebook_mode()
    py.offline.iplot(fig)
    #py.offline.plot(fig, filename='../output/Ce3Dhistogram.html', auto_open=True)
    #pio.write_image(fig, '../output/Ce3Dhistogram.pdf')


# =============================================================================
#             Coincidence Histogram Projections (Front, Top, Side)
# =============================================================================

def ce_projections_plot(df, bus_start, bus_stop, title, norm=1):
    """
    Histograms the hitposition, histogrammed over the three projections of the
    detector (Front, Top and Side).

    Args:
        df (DataFrame): Clustered events

    Returns:
        fig (Figure): Figure containing three 2D coincidences histograms, one
                      for each projection (Front, Top, Side)
        histograms (list): List containing three elements with a
                           2D numpy array 2D matrix each. Each matrix contains
                           the histogram from a specfic projection (front, top,
                           side).
    """
    # Ensure we only plot coincident events
    df = df[(df.wch != -1) & (df.gch != -1)]
    # Define figure and set figure properties
    fig = plt.figure()
    fig.suptitle('Projections - %s' % title)
    fig.set_figheight(4)
    fig.set_figwidth(14)
    # Calculate colorbar limits
    if df.shape[0] != 0:
        vmin = 1
        vmax = df.shape[0] // 20 + 5
    else:
        vmin = 1
        vmax = 1
    # Plot
    wChs, gChs, Buses = df.wch, df.gch, df.bus
    plt.subplot(1, 3, 1)
    h_front = plot_front(wChs, gChs, Buses, bus_start, bus_stop,
                         None, None,
                         #5e-4, 7e-3,
                         norm)
    plt.subplot(1, 3, 2)
    h_top = plot_top(wChs, gChs, Buses, bus_start, bus_stop,
                     None, None,
                     #1e-3, 2e-2,
                     norm)
    plt.subplot(1, 3, 3)
    h_side = plot_side(wChs, gChs, Buses,
                       None, None,
                       #4e-5, 2e-3,
                       norm)
    # Collect all histograms and tighted layout
    plt.tight_layout()
    histograms = [h_front, h_top, h_side]
    return fig, histograms


# =============================================================================
#                               Multiplicity
# =============================================================================

def multiplicity_plot(df, bus_start, bus_stop, title):
    """
    Histograms multiplicity of wires versus grids in the clustered neutron
    events.

    Args:
        df (DataFrame): Clustered events

    Returns:
        fig (Figure): Figure containing nine 2D coincidences histograms, one
                      for each bus.

    """
    def plot_multiplicity_bus(df, bus, vmin, vmax):
        # Plot data
        plt.hist2d(df.wm, df.gm, bins=[80, 40], range=[[0, 80], [0, 40]],
                   norm=LogNorm(), vmin=vmin, vmax=vmax, cmap='jet')
        plt.xlabel('Wire Multiplicity')
        plt.ylabel('Grid Multiplicity')
        plt.colorbar()
        plt.tight_layout()
        plt.title('Bus %d\n(%d events)' % (bus, df.shape[0]))
    # Set limits
    if df.shape[0] != 0:
        vmin = 1
        vmax = df.shape[0] // 9 + 10
    else:
        vmin = 1
        vmax = 1
    # Prepare figure
    fig = plt.figure()
    fig.suptitle('Multiplicity - %s' % title)
    number_detectors = ((bus_stop + 1) - bus_start)//3
    fig.set_figheight(4*number_detectors)
    fig.set_figwidth(14)
    # Iterate through all buses
    for i, bus in enumerate(range(bus_start, bus_stop+1)):
        plt.subplot(number_detectors, 3, i+1)
        df_bus = df[df.bus == bus]
        plot_multiplicity_bus(df_bus, bus, vmin, vmax)
    plt.tight_layout()
    return fig

# =============================================================================
#                                   Timestamp
# =============================================================================


def rate_plot(df, number_bins, label):
    """
    Scatter plot of cluster index vs timestamp, where every 100:th cluster is
    plotted. The color bar shows the summation of wire and grid event, which
    gives an indication of the 'healthiness' of clusters.

    Args:
        df (DataFrame): Clustered events

    Returns:
        fig (Figure): Figure containing the scatter plot of the timestamps
                      from all of the detectors.
    """


    # Prepare figure
    time = (df.time)/(60 ** 2)
    plt.title('Rate vs Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Rate (events/s)')
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Plot
    #plt.hist(Time, histtype='step', bins=number_bins, label=label, zorder=5)
    hist, bin_edges = np.histogram(time, bins=number_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    delta_t = (60 ** 2) * (bin_centers[1] - bin_centers[0])
    plt.plot(bin_centers, hist/delta_t, label=label, zorder=5)


# =============================================================================
#                              HELPER FUNCTIONS
# =============================================================================

def plot_front(wChs, gChs, Buses, bus_start, bus_stop, vmin=None, vmax=None,
               norm=1):
    weights = np.ones(wChs.shape[0]) * norm
    rows = ((bus_stop + 1) - bus_start) * 4
    h_front, *_ = plt.hist2d((wChs + (80*Buses)) // 20, gChs, bins=[rows, 40],
                             range=[[bus_start*4-0.5, (bus_stop+1)*4-0.5], [79.5, 119.5]],
                             norm=LogNorm(),
                             vmin=vmin, vmax=vmax,
                             cmap='jet',
                             weights=weights)
    plt.title('Front view')
    plt.xlabel('Row')
    plt.ylabel('Grid')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    return h_front

def plot_top(wChs, gChs, Buses, bus_start, bus_stop, vmin=None, vmax=None,
             norm=1):
    weights = np.ones(wChs.shape[0]) * norm
    rows = ((bus_stop + 1) - bus_start) * 4
    h_top, *_ = plt.hist2d((wChs + (80*Buses)) // 20, wChs % 20, bins=[rows, 20],
                           range=[[bus_start*4-0.5, (bus_stop+1)*4-0.5], [-0.5, 19.5]],
                           norm=LogNorm(),
                           vmin=vmin, vmax=vmax,
                           cmap='jet',
                           weights=weights)
    plt.title('Top view')
    plt.xlabel('Row')
    plt.ylabel('Layer')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    return h_top

def plot_side(wChs, gChs, Buses, vmin=None, vmax=None, norm=1):
    weights = np.ones(wChs.shape[0]) * norm
    h_side, *_ = plt.hist2d(wChs % 20, gChs, bins=[20, 40],
                            range=[[-0.5, 19.5], [79.5, 119.5]], norm=LogNorm(),
                            vmin=vmin, vmax=vmax,
                            cmap='jet',
                            weights=weights)
    plt.title('Side view')
    plt.xlabel('Layer')
    plt.ylabel('Grid')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    return h_side
