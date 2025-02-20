import os
import requests
import zipfile
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

# Nextcloud direct download link
DATA_URL = "https://kingsx.cs.uni-saarland.de/index.php/s/KFrpMwCfJtaLpX3/download"


def download_data():
    """Downloads and extracts dataset if not already present."""
    data_folder = "data"
    zip_file = "data.zip"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.exists(zip_file):
        st.info("Downloading dataset... This may take a few moments.")
        response = requests.get(DATA_URL, stream=True)

        with open(zip_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.success("Download complete. Extracting files...")

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

        st.success("Extraction complete.")


# Run the download function before launching the app
download_data()


@st.cache_data
def load_data():
    """Loads AFM dataset from extracted files."""
    data_folder = "data"

    # Dynamically detect extracted files
    hname = None
    fname = None
    for file in os.listdir(data_folder):
        if file.endswith(".heights.npy"):
            hname = os.path.join(data_folder, file)
        elif file.endswith(".data.pickled"):
            fname = os.path.join(data_folder, file)

    if not hname or not fname:
        st.error("Data files not found. Please check the download link.")
        return None, None, None, None

    st.info("Loading data...")
    H = np.load(hname)
    m, n = H.shape

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # Estimate slopes
    slope_est = {}
    for point, curve in data.items():
        s, i, j = point
        slope_est[point] = estimate_slope(curve, s)

    nseries = 1 + max(s for s, i, j in data.keys())
    slope_heatmaps = []
    for s in range(nseries):
        M = np.array([[(slope_est[s, i, j][0]) for j in range(n)] for i in range(m)], dtype=np.float64)
        slope_heatmaps.append(M)

    st.success("Data loaded successfully!")
    return H, data, slope_est, slope_heatmaps


def do_plot(point, curve, slope=None, anchors=None):
    """Plots force-distance curve with estimated slope."""
    s, i, j = point
    d, f = curve

    fig = plt.figure(figsize=[10, 6])
    plt.xlabel("Distance (m)")
    plt.ylabel("Force (N)")
    plt.scatter(d, f, s=1, label=f'Data at ({i}, {j})')

    if slope is not None and anchors is not None:
        anchor1, anchor2 = anchors
        plt.axline(anchor1, slope=slope, color='red', linestyle='--', label=f'{slope:.4g} N/m')
        plt.plot([anchor1[0]], [anchor1[1]], 'rx')
        plt.plot([anchor2[0]], [anchor2[1]], 'rx')

    plt.legend()
    return fig


def estimate_slope(curve, s):
    """Estimates slope of the AFM force-distance curve."""
    d, f = curve
    if s == 0:
        d, f = d[::-1], f[::-1]  # Reverse for series-0 spectra

    window = 20
    slopes = []
    indices = range(0, len(d) - window + 1, 30)

    for start in indices:
        end = start + window
        x = d[start:end]
        y = f[start:end]
        coefficients = np.polyfit(x, y, 1)
        slopes.append(coefficients[0])

    mid = len(d) // 2
    quad = mid // 4
    half_slopes_sorted = np.argsort(-np.abs(slopes))

    largest = None
    for i in half_slopes_sorted:
        if indices[i] < mid and indices[i] > quad:
            largest = i
            break

    if largest is None:
        for i in half_slopes_sorted:
            if indices[i] < mid + quad and indices[i] > quad:
                largest = i
                break

    start = indices[largest]
    end = start + window
    d1, f1 = d[start], f[start]
    d2, f2 = d[end - 1], f[end - 1]

    return (f2 - f1) / (d2 - d1), ((d1, f1), (d2, f2)), None


# Load data
H, S, slope_est, slope_heatmaps = load_data()
if H is None:
    st.stop()

m, n = H.shape
nseries = len(slope_heatmaps)

# Sidebar options
st.sidebar.title("AFM Data Explorer")
series = st.sidebar.radio("Series:", options=[0, 1], format_func=lambda x: "0 [Push]" if x == 0 else "1 [Retract]",
                          index=0)
i = st.sidebar.slider("Vertical Coordinate (i)", 0, m - 1, 0)
j = st.sidebar.slider("Horizontal Coordinate (j)", 0, n - 1, 0)

choice = (series, i, j)
force_distance_data = S[choice]
calculated_slope, anchor_points, _ = slope_est[choice]
force_plot = do_plot(choice, force_distance_data, calculated_slope, anchor_points)

st.pyplot(force_plot)

# Heatmap Visualizations
st.subheader("Slope Heatmap")
slope_heatmap = px.imshow(slope_heatmaps[series], color_continuous_scale='turbo', origin='lower', zmin=-0.013,
                          zmax=-0.003)
slope_heatmap.update_yaxes(autorange="reversed")
st.plotly_chart(slope_heatmap)

st.subheader("Height Heatmap")
height_heatmap = px.imshow(H, color_continuous_scale='turbo', origin='lower')
height_heatmap.update_yaxes(autorange="reversed")
st.plotly_chart(height_heatmap)
