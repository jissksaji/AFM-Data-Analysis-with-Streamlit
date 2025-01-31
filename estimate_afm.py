from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sys

def extract_data_points(fname):
    # return a dict called 'data', such that
    # data[s, I, j] = (d, f),
    # where s is the series, I, j are the point coordinates;
    # d is a list containing measured distances,
    # f is a list containing measured forces.
    data = {}
    with open(fname, 'rt') as ftext:
        temp_index = []
        d = defaultdict(list)
        temp = []
        force = []
        distance = []
        series = -1
        i, j = 0, 0
        change = False
        prev_s, s = -1, -1
        key = None

        for line in ftext:
            line = line.strip()
            indices = []
            if line.startswith('# index'):
                s = int(line.split()[2])
                change = True
            if line.startswith('# iIndex'):
                i = int(line.split()[2])
            if line.startswith('# jIndex'):
                j = int(line.split()[2])
                key = (s, i, j)
                data[key] = ([], [])
            if change and prev_s == -1 and s != -1:
                prev_s = 0
                s = 0
                change = False
            if change and prev_s == 0 and s != -1:
                prev_s = -1
                s = 1
                change = False
            temp_index.append(indices)
            if not line.startswith('#') and key:
                line = line.split()
                data[key][0].append(float(line[0]))
                data[key][1].append(float(line[1]))
    return data


def calculate_slopes(data):
    window = 20
    slope_dict = {}
    for key, (distance, force) in data.items():
        if distance[0] > distance[-1]:
            distance = distance[::-1]
            force = force[::-1]
        slopes = []
        indices = range(0, len(distance) - window + 1, 30)
        for start in indices:
            end = start + window
            x = distance[start:end]
            y = force[start:end]
            coefficients = np.polyfit(x, y,1)
            slope = coefficients[0]
            slopes.append(slope)
        mid = len(distance) // 2
        quad = mid // 4
        half_slopes_sorted= np.argsort(-np.abs(slopes))
        for i in half_slopes_sorted:
            if indices[i] < mid and indices[i] > quad:
                largest = i
                break
        start = indices[largest]
        end = start + window
        point1 = (distance[start], force[start])
        point2 = (distance[end - 1], force[end - 1])
        slope_dict[key] = {
            'slope': (point2[1] - point1[1]) / (point2[0] - point1[0]),
            'point1': point1,
            'point2': point2
        }

    return slope_dict

def print_slopes(slope_dict):
    nos = len(slope_dict.keys())
    print(f"#Processing {nos} spectra....")
    for (s, i, j), details in slope_dict.items():
        slope = details['slope']
        print(f"{s} {i} {j} {slope:.5f}")


#data1 = extract_data_points('sample.txt')
#slope_data = calculate_slopes(data1)
#print_slopes(slope_data)

def raw_plot(data, slope_dict, save_prefix, show):
    for key, (distance, force) in data.items():
        if key[0] == 0:
            details = slope_dict.get(key, None)
            point1 = details['point1']
            point2 = details['point2']
            slope = details['slope']
            intercept = point1[1] - slope * point1[0]
            x_intercept = -intercept / slope
            x_extrapolate = np.linspace(min(0, x_intercept), max(x_intercept, max(distance)), 600)
            y_extrapolate = slope * x_extrapolate + intercept
            d_scaled = np.array(distance) * 1e6
            f_scaled = np.array(force) * 1e9
            plt.figure(figsize=(9, 6))
            plt.plot(d_scaled, f_scaled, label='data')
            plt.scatter([point1[0] * 1e6, point2[0] * 1e6],
                        [point1[1] * 1e9, point2[1] * 1e9], color='red', marker='x', zorder=10)
            plt.plot(x_extrapolate * 1e6, y_extrapolate * 1e9, 'r--', linewidth=2)
            plt.title(f"push at  (0,  {key[1]})")
            plt.xlabel("Distance (M)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.grid(True)
            plt.xlim(min(d_scaled), max(d_scaled))
            plt.ylim(min(f_scaled), max(f_scaled))
            if save_prefix:
                fname = f"{save_prefix}_series0_i{key[1]}_j{key[2]}.png"
                plt.savefig(fname, dpi=200, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()




#raw_plot(data1, slope_data, save_prefix='series0', show=True)

'''def do_raw_plots(data, show, plotprefix):
    for point, curve in data.items():
        s, i, j = point
        print(f"plotting curve at {point}")
        fname = f'{plotprefix}-{s:01d}-{i:03d}-{j:03d}.png' if plotprefix is not None else None
        raw_plot(point, curve, show=show, save=fname)
'''
def get_argument_parser():
    p = ArgumentParser()
    p.add_argument("--textfile", "-t", required=True,
        help="name of the data file containing AFM curves for many points")
    p.add_argument("--first", type=int,
        help="number of curves to extract and plot")
    p.add_argument("--plotprefix", default="curve",
        help="non-empty path prefix of plot files (PNGs); do not save plots if not given")
    p.add_argument("--show", action="store_true",
        help="show each plot")
    return p


def main(args):
    print(f"#Parsing {args.textfile}...")
    data = extract_data_points(args.textfile)
    slope_dict = calculate_slopes(data)
    print_slopes(slope_dict)
    if '--plotprefix' in sys.argv:
        for key in data.keys():
            raw_plot({key: data[key]}, slope_dict, save_prefix=args.plotprefix, show=True)

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)

