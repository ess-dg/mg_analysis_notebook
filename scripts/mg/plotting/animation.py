#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animation3D.py: Helper functions for handling of paths and folders.
"""

import plotly as py
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import os
import imageio
import shutil

from mg.helper_functions.mapping import create_mapping
from mg.helper_functions.border_lines import initiate_detector_border_lines
from mg.helper_functions.misc import mkdir_p

from mg.plotting.basic_plotting import plot_front, plot_top, plot_side
from mg.plotting.basic_plotting import tof_histogram
#from mg.plotting.basic_plotting import timestamp_plot


# =============================================================================
#                         3D Animation - Time sweep
# =============================================================================

def Animation_3D_plot(ce_full, origin_voxel):
    # Filters
    ce = ce_full[ce_full.Bus == 1]
    ce = ce[(ce.wCh != -1) & (ce.gCh != -1)]
    # Time intervals
    iter_start = 0
    iter_stop = 60
    step = 1
    conversion_factor = 1/(62.5e-9)
    # Mapping and Borders
    mapping = create_mapping(origin_voxel)
    borderline_traces = initiate_detector_border_lines(mapping)
    # Plotting
    min_count = 0
    max_count = np.inf
    cmin = 1
    cmax = 10
    # Storage
    dir_name = os.path.dirname(__file__)
    temp_animation_folder = os.path.join(dir_name, '../temp_animation_folder/')
    output_path = os.path.join(dir_name, '../3D_animation.gif')
    mkdir_p(temp_animation_folder)
    # Iteration
    delimiters = np.arange(iter_start, iter_stop, step)
    for i, delimiter in enumerate(delimiters):
        # Get current limits and filter data
        start = delimiter * conversion_factor
        stop = (delimiter + step) * conversion_factor
        ce_step = ce[(ce.Time >= start) & (ce.Time <= stop)]
        # Plot
        traces = []
        hist = get_3D_histogram(ce_step, mapping, min_count, max_count)
        MG_3D_trace = get_MG_3D_trace(hist, cmin, cmax)
        traces.append(MG_3D_trace)
        traces.extend(borderline_traces)
        fig = create_fig(traces, start, stop)
        # Save
        path = temp_animation_folder + '%d.png' % i
        pio.write_image(fig, path)
    # Animate
    images = []
    files = os.listdir(temp_animation_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' and file != '.gitignore']
    for file in sorted(files, key=int):
        images.append(imageio.imread(temp_animation_folder + file + '.png'))
    imageio.mimsave(output_path, images)
    shutil.rmtree(temp_animation_folder, ignore_errors=True)



# =============================================================================
#                                LAMBDA SWEEP
# =============================================================================

def Lambda_Sweep_Animation(ce, number_bins, origin_voxel, bus_start, bus_stop):
    def meV_to_A(energy):
        return np.sqrt(81.81/energy)

    def A_to_meV(wavelength):
        return (81.81/(wavelength ** 2))
    # Filters
    ce = ce[(ce.wCh != -1) & (ce.gCh != -1)]
    # Storage
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../temp_folder/')
    output_path = os.path.join(dir_name, '../../../../output/lambda_sweep.gif')
    mkdir_p(temp_folder)
    # Calculate energies
    energy = calculate_energy(ce, origin_voxel)
    landa = meV_to_A(energy)
    # Define intervals
    iter_start = 4.3
    iter_stop = 4.75
    step = 0.001
    delimiters = np.arange(iter_start, iter_stop, step)
    # Iteration
    vmin = 1
    vmax = 1e3
    for i, delimiter in enumerate(delimiters):
        start = delimiter
        stop = delimiter + step
        ce_temp = ce[(landa >= start) & (landa <= stop)]
        wChs, gChs, Buses = ce_temp.wCh, ce_temp.gCh, ce_temp.Bus
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(14)
        plt.subplot2grid((2, 3), (0, 0), colspan=1)
        plot_front(wChs, gChs, Buses, bus_start, bus_stop, vmin, vmax)
        plt.subplot2grid((2, 3), (0, 1), colspan=1)
        plot_top(wChs, gChs, Buses, bus_start, bus_stop, vmin, vmax)
        plt.subplot2grid((2, 3), (0, 2), colspan=1)
        plot_side(wChs, gChs, Buses, vmin, vmax)
        plt.subplot2grid((2, 3), (1, 0), colspan=3)
        energy_plot(ce, origin_voxel, number_bins, iter_start, iter_stop)
        plt.axvline(x=(start+stop)/2, color='red', linewidth=2, alpha=0.8, zorder=5)
        plt.tight_layout()
        fig.savefig(temp_folder + '%d.png' % i)
        plt.close(fig)
    # Animate
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' and file != '.gitignore']
    for file in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + file + '.png'))
    imageio.mimsave(output_path, images)
    shutil.rmtree(temp_folder, ignore_errors=True)


# =============================================================================
#                                TIME SWEEP
# =============================================================================

def Time_Sweep_Animation(ce, number_bins, origin_voxel, bus_start, bus_stop):
    # Filters
    ce = ce[(ce.wCh != -1) & (ce.gCh != -1)]
    # Storage
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../temp_folder/')
    output_path = os.path.join(dir_name, '../../../../output/time_sweep.gif')
    mkdir_p(temp_folder)
    # Define intervals and extract Times (UNIT IS IN SECONDS)
    iter_start = (12 * 60 ** 2)
    iter_stop = (12 * 60 ** 2) + 2
    step = 0.05
    delimiters = np.arange(iter_start, iter_stop, step)
    Time = ce.Time.values*62.5e-9  # UNIT [SECONDS]
    unit = 's'
    # Iteration
    vmin = 1
    vmax = 1e1
    for i, delimiter in enumerate(delimiters):
        start = delimiter
        stop = delimiter + step
        ce_temp = ce[(Time >= start) & (Time <= stop)]
        wChs, gChs, Buses = ce_temp.wCh, ce_temp.gCh, ce_temp.Bus
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(14)
        plt.subplot2grid((2, 3), (0, 0), colspan=1)
        plot_front(wChs, gChs, Buses, bus_start, bus_stop, vmin, vmax)
        plt.subplot2grid((2, 3), (0, 1), colspan=1)
        plot_top(wChs, gChs, Buses, bus_start, bus_stop, vmin, vmax)
        plt.subplot2grid((2, 3), (0, 2), colspan=1)
        plot_side(wChs, gChs, Buses, vmin, vmax)
        plt.subplot2grid((2, 3), (1, 0), colspan=3)
        timestamp_plot(Time[(Time >= iter_start) & (Time <= iter_stop)], number_bins, unit)
        plt.axvline(x=(start+stop)/2, color='red', linewidth=2, alpha=0.8, zorder=5)
        plt.tight_layout()
        fig.savefig(temp_folder + '%d.png' % i)
        plt.close(fig)
    # Animate
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' and file != '.gitignore']
    for file in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + file + '.png'))
    imageio.mimsave(output_path, images)
    shutil.rmtree(temp_folder, ignore_errors=True)


# =============================================================================
#                                TOF SWEEP
# =============================================================================

def ToF_Sweep_Animation(ce, number_bins, bus_start, bus_stop):
    # Filters
    ce = ce[(ce.wCh != -1) & (ce.gCh != -1)]
    # Storage
    dir_name = os.path.dirname(__file__)
    temp_folder = os.path.join(dir_name, '../temp_folder/')
    output_path = os.path.join(dir_name, '../../../../output/ToF_sweep.gif')
    mkdir_p(temp_folder)
    # Declare parameters
    time_offset = (0.6e-3) * 1e6
    period_time = (1/14) * 1e6
    # Calculate ToF
    ToF = (ce.ToF.values * 62.5e-9 * 1e6 + time_offset) % period_time
    # Define intervals
    iter_start = 0
    iter_stop = 70000
    step = 500
    delimiters = np.arange(iter_start, iter_stop, step)
    # Iteration
    vmin = 1
    vmax = 1e3
    for i, delimiter in enumerate(delimiters):
        start = delimiter
        stop = delimiter + step
        ce_temp = ce[(ToF >= start) & (ToF <= stop)]
        wChs, gChs, Buses = ce_temp.wCh, ce_temp.gCh, ce_temp.Bus
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(14)
        plt.subplot2grid((2, 3), (0, 0), colspan=1)
        plot_front(wChs, gChs, Buses, bus_start, bus_stop, vmin, vmax)
        plt.subplot2grid((2, 3), (0, 1), colspan=1)
        plot_top(wChs, gChs, Buses, bus_start, bus_stop, vmin, vmax)
        plt.subplot2grid((2, 3), (0, 2), colspan=1)
        plot_side(wChs, gChs, Buses, vmin, vmax)
        plt.subplot2grid((2, 3), (1, 0), colspan=3)
        ToF_histogram(ce[(ToF >= iter_start) & (ToF <= iter_stop)], number_bins)
        plt.axvline(x=(start+stop)/2, color='red', linewidth=2, alpha=0.8, zorder=5)
        plt.tight_layout()
        fig.savefig(temp_folder + '%d.png' % i)
        plt.close(fig)
    # Animate
    images = []
    files = os.listdir(temp_folder)
    files = [file[:-4] for file in files if file[-9:] != '.DS_Store' and file != '.gitignore']
    for file in sorted(files, key=int):
        images.append(imageio.imread(temp_folder + file + '.png'))
    imageio.mimsave(output_path, images)
    shutil.rmtree(temp_folder, ignore_errors=True)


# =============================================================================
#                                Helper Functions
# =============================================================================

def get_3D_histogram(ce, mapping, min_count, max_count):
    # Calculate 3D histogram
    H, edges = np.histogramdd(ce[['wCh', 'gCh', 'Bus']].values,
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
    return hist

def get_MG_3D_trace(hist, cmin=None, cmax=None):
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
                                           cmin=cmin,
                                           cmax=cmax
                                           ),
                               name='Multi-Grid',
                               scene='scene1'
                               )
    return MG_3D_trace

def create_fig(traces, start, stop):
    # Introduce figure and put everything together
    fig = py.tools.make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])
    # Insert vessel borders
    for trace in traces:
        fig.append_trace(trace, 1, 1)
    # Assign layout with axis labels, title and camera angle
    a = 0.6
    camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=-3, y=-6, z=-2),
                  eye=dict(x=1*a, y=1*a, z=1*a))
    scene=dict(camera=camera, #the default values are 1.25, 1.25, 1.25
           xaxis=dict(),
           yaxis=dict(),
           zaxis=dict(),
           aspectmode='data')
    fig['layout']['scene1']['xaxis'].update(title='x [m]')
    fig['layout']['scene1']['yaxis'].update(title='y [m]')
    fig['layout']['scene1']['zaxis'].update(title='z [m]')
    fig['layout'].update(title='Start: %.1f, Stop: %.1f' % (start, stop))
    fig['layout']['scene1'].update(scene)
    fig.layout.showlegend = False
    return fig
