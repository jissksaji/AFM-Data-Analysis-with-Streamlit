"""
1. Rename this file to app.py
2. Download the preprocessed data files 'afm.heights.npy', 'afm.data.pickled' at:
   https://kingsx.cs.uni-saarland.de/index.php/s/KFrpMwCfJtaLpX3
3. streamlit run app.py -- afm
   This will take time to load the data and ESTIMATE the slopes with your code
   to be written here in function estimate_slope().
   Make sure that your code is efficient enough to run in a few seconds.
   Otherwise your application will seem to load forever.
"""


from argparse import ArgumentParser
import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

# You can use plotly instead of pyplot for nicer (and possibly interactive) plots.
# However, they must be installed separately using mamba or pip:

# import plotly.express as px  # mamba install plotly
# from streamlit_plotly_events import plotly_events  # pip install streamlit_plotly_events


@st.cache_data
def load_data(prefix):
    print(f"- Loading {prefix}{{.data.pickled,.heights.npy}}")
    hname = f"{prefix}.heights.npy"
    H = np.load(hname)
    m, n = H.shape
    fname = f"{prefix}.data.pickled"
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # Estimate all slopes
    slope_est = dict()  # maps tuple (s, i, j) to tuple (slope, anchors, info)
    for point, curve in data.items():
        s, i, j = point
        slope_est[point] = estimate_slope(curve, s)
    nseries = 1 + max(s for s, i, j in data.keys())
    slope_heatmaps = []
    for s in range(nseries):
        M = np.array([[(slope_est[s, i, j][0]) for j in range(n)] for i in range(m)], dtype=np.float64)
        slope_heatmaps.append(M)
    return H, data, slope_est, slope_heatmaps


def do_plot(point, curve, slope=None, anchors=None):
    """plot one distance-force curve with estimated slope"""
    s, i, j = point
    d, f = curve
    fig = plt.figure(figsize=[10, 6])
    plt.xlabel("distance (m)")
    plt.ylabel("force (N)")
    mode = 'push' if s == 0 else 'retract'
    plt.title(f"{mode} at ({i}, {j});  number of records: {len(d)}")
    label = f'data: {mode} at {(i, j)}'
    plt.scatter(d, f, s=1, label=label)
    plt.grid()
    if slope is not None and anchors is not None:
        anchor0, anchor1 = anchors[0], anchors[1]
        plt.axline(anchor0, slope=slope, color='red', linestyle='--', label=f'{slope:.4g} N/m')
        plt.plot([anchor0[0]], [anchor0[1]], 'rx')
        plt.plot([anchor1[0]], [anchor1[1]], 'rx')
    plt.legend()
    return fig


def estimate_slope(curve, s, nan=float("nan")):
    d, f = curve
    if s == 0:
        d, f = d[::-1], f[::-1]  # reverse d and f for series-0 spectra !
    # d is now increasing, f (on the left side) is decreasing.
    half_force = f[-1] + (f[0] - f[-1]) / 2
    window = 20
    slopes = []
    indices = range(0, len(d) - window + 1, 30)

    for start in indices:
        end = start + window
        x = d[start:end]
        y = f[start:end]
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        slopes.append(slope)

    mid = len(d) // 2
    quad = mid // 4
    half_slopes_sorted = np.argsort(-np.abs(slopes))
    largest = None
    for i in half_slopes_sorted:
        if indices[i] < mid and indices[i] > quad and f[indices[i]] < half_force:
            largest = i
            break
    if largest is None:
        for i in half_slopes_sorted:
            if indices[i] < mid+quad and indices[i] > quad and f[indices[i]] < half_force:
                largest = i
                break

    start = indices[largest]
    end = start + window
    d1, f1 = d[start], f[start]
    d2, f2 = d[end - 1], f[end - 1]
    anchor1 = (d1, f1)
    anchor2 = (d2, f2)
    slope = (f2 - f1) / (d2 - d1)
    # You must return a tuple (slope, anchors, info), where:
    # - slope is the estimated slope (a single float number)
    # - anchors is a 2-tuple (anchor1, anchor2), where
    #   - anchor1 is a point (d1, f1) of distance, force
    #   - anchor2 is a point (d2, f2) like anchor1,
    #   - the slope has been estimated between distances d1 and d2.
    #   - (These two points will be plotted as red x in the do_plot function!)
    # - info can be anything that you want to return as diagnostic information.
    #   It is currently ignored in this program, but it can help you debug.

    anchors = (anchor1, anchor2)  # anchor1 = (some_d, some_f), same for anchor2
    info = None  # can by anything you want to return in addition
    return (slope, anchors, info)


# MAIN script

p = ArgumentParser()
p.add_argument("prefix",
    help="common path prefix for spectra (.data.pickled) and heights (.heights.npy)")
args = p.parse_args()
prefix = args.prefix

st.sidebar.title("AFM Data Explorer")
st.sidebar.write(f"Path prefix:\n'{prefix}'")
H, S, slope_est, slope_heatmaps = load_data(prefix)  # cached
m, n = H.shape
nseries = len(slope_heatmaps)

# At this point, we have the following:
# - H is a m * n numpy array (float64) of measured heights.
# - S is a Python dict of measurements, indexed like S[s, i, j] = (d, f)
#   where s: series (0 or 1), i: vertical coordinate, j: horizontal coordinate
#   d: distance values, f: force values
# -

# Here, create a streamlit interface like in the example shown in the slides.
# It does not have to look exactly like this, but should offer the same actions:
# - Pick s, i, j of the measurement series to show on the left sidebar.
# - Show overview heatmaps of the heights H and the chosen slopes;
#   either slope_heatmaps[0] for series 0, or slope_heatmaps[1] for series 1.
#   In the example, the color scale (colormap) 'turbo' or 'Turbo' was used.
#   You are free to choose other color scales as long as they make sense.
# - At the bottom, show the selected measurement series S[s,i,j] = (d,f)
#   with d on the x-axis and f on the y-axis.

series = st.sidebar.radio("Series:",options=[0, 1],format_func=lambda x: "0 [push]" if x == 0 else "1 [retract]",index=0)
i = st.sidebar.slider("Coordinate i (vertical)", 0, m - 1, 0)
j = st.sidebar.slider("Coordinate j(horizontal)", 0, n - 1, 0)
choice = (series, i, j)
force_distance_data = S[choice]
calculated_slope, anchor_points, _ = slope_est[choice]
force_plot = do_plot(choice, force_distance_data, calculated_slope, anchor_points)
st.pyplot(force_plot)
st.subheader("Slope")
heatmap_data_slope = slope_heatmaps[series]
slope_1 = px.imshow(heatmap_data_slope,color_continuous_scale='turbo',origin='lower',zmin=-0.013,zmax=-0.003)
slope_1.update_yaxes(autorange="reversed")
st.plotly_chart(slope_1)
st.subheader("Heights")
height = px.imshow(H,color_continuous_scale='turbo',origin='lower')
height.update_yaxes(autorange="reversed")
st.plotly_chart(height)
