#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" basic_plotting.py:
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
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


def phs_1d_plot(events, number_bins, bus, vmin, vmax):
    """
    ...
    ...

    Args:
        events (DataFrame): Individual events
        number_bins (int): Number of bins to use in the histogram

    Yields:
        Figure containing a 1D PHS plot
    """

    plt.hist(events[events.ch <= 79].adc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 4400], label='Wires', color='blue')
    plt.hist(events[events.ch >= 80].adc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 4400], label='Grids', color='red')
    plt.title('Bus: %d' % bus)
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.legend()

def phs_clusters_1d_plot(clusters, clusters_uf, number_bins, bus, duration):
    # Clusters filtered
    plt.hist(clusters.wadc, bins=number_bins, histtype='step',
             zorder=5,
             range=[0, 8000],
             label='Wires (filtered)', color='blue',
             weights=(1/duration)*np.ones(len(clusters.wadc)))
    plt.hist(clusters.gadc, bins=number_bins, histtype='step',
             zorder=5,
             range=[0, 8000],
             label='Grids (filtered)', color='red',
             weights=(1/duration)*np.ones(len(clusters.gadc)))
    # Clusters unfiltered
    hist_w, bins_w, __ = plt.hist(clusters_uf.wadc, bins=number_bins, histtype='step',
                              zorder=5,
                              range=[0, 8000],
                              label='Wires', color='cyan',
                              weights=(1/duration)*np.ones(len(clusters_uf.wadc)))
    hist_g, bins_g, __ = plt.hist(clusters_uf.gadc, bins=number_bins, histtype='step',
                                  zorder=5,
                                  range=[0, 8000],
                                  label='Grids', color='magenta',
                                  weights=(1/duration)*np.ones(len(clusters_uf.gadc)))
    plt.title('Bus: %d' % bus)
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts/s')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.legend()
    # Save histograms
    bins_w_c = 0.5 * (bins_w[1:] + bins_w[:-1])
    bins_g_c = 0.5 * (bins_g[1:] + bins_g[:-1])
    np.savetxt('../output/seq_phs_unfiltered_wires_bus_%d.txt' % bus,
               np.transpose(np.array([bins_w_c, hist_w])),
               delimiter=",",header='bins, hist (counts/s)')
    np.savetxt('../output/seq_phs_unfiltered_grids_bus_%d.txt' % bus,
               np.transpose(np.array([bins_g_c, hist_g])),
               delimiter=",",header='bins, hist (counts/s)')

# =============================================================================
#                                   PHS (2D)
# =============================================================================

def phs_2d_plot(events, bus, vmin, vmax):
    """
    Histograms the ADC-values from each channel individually and summarises it
    in a 2D histogram plot, where the color scale indicates number of counts.
    Each bus is presented in an individual plot.

    Args:
        events (DataFrame): Individual events

    Returns:
        fig (Figure): Figure containing 2D PHS plot
    """
    plt.xlabel('Channel')
    plt.ylabel('Charge (ADC channels)')
    plt.title('Bus: %d' % bus)
    bins = [120, 120]
    if events.shape[0] > 1:
        plt.hist2d(events.ch, events.adc, bins=bins, norm=LogNorm(),
                   range=[[-0.5, 119.5], [0, 4400]],
                   vmin=vmin, vmax=vmax,
                   cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Counts')


# =============================================================================
#                           PHS (Wires vs Grids)
# =============================================================================


def clusters_phs_plot(clusters, bus, duration, vmin, vmax):
    """
    Histograms ADC charge from wires vs grids, one for each bus, showing the
    relationship between charge collected by wires and charge collected by
    grids. In the ideal case there should be linear relationship between these
    two quantities.

    Args:
        clusters (DataFrame): Clustered events

    Returns:

    """

    plt.xlabel('Charge wires (ADC channels)')
    plt.ylabel('Charge grids (ADC channels)')
    plt.title('Bus: %d' % bus)
    bins = [200, 200]
    ADC_range = [[0, 10000], [0, 10000]]
    plt.hist2d(clusters.wadc, clusters.gadc, bins=bins, norm=LogNorm(),
               range=ADC_range,
               vmin=vmin, vmax=vmax, cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wadc)))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')


# =============================================================================
#                          Coincidence Histogram (2D)
# =============================================================================

def clusters_2d_plot(clusters, title, vmin, vmax, duration):
    """

    """

    plt.hist2d(clusters.wch, clusters.gch, bins=[80, 40],
               range=[[-0.5, 79.5], [79.5, 119.5]],
               vmin=vmin, vmax=vmax,
               norm=LogNorm(), cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    plt.xlabel('Wire (Channel number)')
    plt.ylabel('Grid (Channel number)')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')


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
                                           color=hist[3],#np.log10(hist[3]),
                                           colorscale='Jet',
                                           opacity=1,
                                           colorbar=dict(thickness=20,
                                                         title='counts'#'log10(counts)'
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
    #fig['layout'].update(title='Coincidences (3D) - %s' % title)
    fig['layout']['scene1'].update(scene)
    fig.layout.showlegend = False
    py.offline.init_notebook_mode()
    py.offline.iplot(fig)
    #py.offline.plot(fig, filename='../output/Ce3Dhistogram.html', auto_open=True)
    pio.write_image(fig, '../output/Ce3Dhistogram.pdf')


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
                         #1e4, 1e6,
                         norm)
    plt.subplot(1, 3, 2)
    h_top = plot_top(wChs, gChs, Buses, bus_start, bus_stop,
                     None, None,
                     #1e4, 1e6,
                     norm)
    plt.subplot(1, 3, 3)
    h_side = plot_side(wChs, gChs, Buses,
                       None, None,
                      # 1e4, 5e5,
                       norm)
    # Collect all histograms and tighted layout
    plt.tight_layout()
    #histograms = [h_front, h_top, h_side]
    #return histograms


# =============================================================================
#                               Multiplicity
# =============================================================================

def multiplicity_plot(clusters_uf, bus, duration):
    """


    """

    # Plot data
    plt.hist2d(clusters_uf.wm, clusters_uf.gm,
               bins=[80, 40], range=[[0, 80], [0, 40]],
               norm=LogNorm(),
               #vmin=vmin, vmax=vmax,
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters_uf.wm)))
    plt.xlabel('Wire Multiplicity')
    plt.ylabel('Grid Multiplicity')
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.title('Bus: %d' % bus)
    plt.tight_layout()


def multiplicity_plot_perc(clusters_uf, bus, duration, vmin=None, vmax=None):
    # Declare parameters
    m_range = [0, 10, 0, 10]
    # Plot data
    hist, xbins, ybins, im = plt.hist2d(clusters_uf.wm, clusters_uf.gm,
                                        bins=[m_range[1]-m_range[0]+1,
                                              m_range[3]-m_range[2]+1],
                                        range=[[m_range[0], m_range[1]+1],
                                               [m_range[2], m_range[3]+1]],
                                        norm=LogNorm(),
                                        vmin=vmin, vmax=vmax,
                                        cmap='jet',
                                        weights=(1/duration)*np.ones(len(clusters_uf.wm)))
    # Iterate through all squares and write percentages
    tot = clusters_uf.shape[0] * (1/duration)
    font_size = 8
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5, ybins[i]+0.5,
                                '%.d%%' % (100*(hist[j, i]/tot)),
                                color="w", ha="center", va="center",
                                fontweight="bold", fontsize=font_size)
                text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                           foreground='black'),
                                       path_effects.Normal()])
    # Set ticks on axis
    ticks_x = np.arange(m_range[0], m_range[1]+1, 1)
    locs_x = np.arange(m_range[0] + 0.5, m_range[1]+1.5, 1)
    ticks_y = np.arange(m_range[2], m_range[3]+1, 1)
    locs_y = np.arange(m_range[2] + 0.5, m_range[3]+1.5, 1)
    plt.xticks(locs_x, ticks_x)
    plt.yticks(locs_y, ticks_y)
    plt.xlabel("Wire multiplicity")
    plt.ylabel("Grid multiplicity")
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.title('Bus: %d' % bus)
    plt.tight_layout()


# =============================================================================
#                                   Timestamp
# =============================================================================


def rate_plot(clusters, number_bins, bus, area):
    """

    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('Time (hours)')
    plt.ylabel('Rate (events/s)')
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Plot
    time = (clusters.time * 62.5e-9)/(60 ** 2)
    hist, bin_edges = np.histogram(time, bins=number_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    delta_t = (60 ** 2) * (bin_centers[1] - bin_centers[0])
    plt.plot(bin_centers, (hist/delta_t), marker='o', linestyle='--',
            zorder=5, color='black')
    return min((hist/delta_t)), max((hist/delta_t))


# =============================================================================
#                               TIME-OF-FLIGHT
# =============================================================================



def tof_histogram(clusters, number_bins, bus):
    """
    Histograms the ToF values in the clustered data.

    Args:
        df (DataFrame): Clustered events
        number_bins (int): The number of bins to histogram the data into

    Yields:
        Figure containing the ToF histogram, this can then be used in overlay
        or in sub plots.
    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('tof ($\mu$s)')
    plt.ylabel('Counts')
    #plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Histogram data
    plt.hist(clusters.tof * 62.5e-9 * 1e6, bins=number_bins, zorder=4,
             histtype='step', color='black')






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
