
from itertools import islice
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt


def extract_data_points(fname):
    # return a dict called 'data', such that
    # data[s, i, j] = (d, f),
    # where s is the series, i, j are the point coordinates;
    # d is a numpy array containing measured distances,
    # f is a numpy array containing measured forces.
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
        for line in ftext:
            line = line.strip()
            indices = []
            if line.startswith('# index'):
                s = int(line.strip().split()[2])
                change = True
            if line.startswith('# iIndex'):
                i = int(line.strip().split()[2])
            if line.startswith('# jIndex'):
                j = int(line.strip().split()[2])
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

        for key in data:
            data[key] = (np.array(data[key][0]), np.array(data[key][1]))
    return data


def raw_plot(point, curve, save=None, show=True):
    d, f = curve
    plt.figure(figsize=[9, 6])
    s, I, j = point
    d_scaled = d * 1e6
    f_scaled = f * 1e9
    plt.plot(d_scaled, f_scaled, label='Force vs Distance', color='blue')
    plt.title(f' Force vs Distance at Series {s}, I = {I}, j = {j}')
    plt.xlabel('Distance(m)')
    plt.ylabel('Force(N)')
    plt.grid()
    if save is not None:
        plt.savefig(save, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def do_raw_plots(data, show, plotprefix):
    for point, curve in data.items():
        s, i, j = point
        print(f"plotting curve at {point}")
        fname = f'{plotprefix}-{s:01d}-{i:03d}-{j:03d}.png' if plotprefix is not None else None
        raw_plot(point, curve, show=show, save=fname)


def main(args):
    fname = args.textfile
    print(f"parsing {fname}...")
    full_data = extract_data_points(fname)
    if args.first is not None:
        data = dict((k, v) for k, v in islice(full_data.items(), args.first))
    else:
        data = full_data
    do_raw_plots(data, args.show, args.plotprefix)


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


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
