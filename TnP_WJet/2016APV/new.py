import ROOT as r
import uproot
import json
import os
import matplotlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.offsetbox import AnchoredText
import argparse
from uncertainties import unumpy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convTH1(TH1):
    vals = TH1.values()
    edges = TH1.axes[0].edges()
    variances = TH1.variances()
    vals = vals * np.diff(edges)
    variances = variances * np.diff(edges)
    return vals, edges, variances


def covnTGA(tgasym):
    _x, _y = tgasym.values(axis='both')
    _xerrlo, _xerrhi = tgasym.errors("low", axis='x'), tgasym.errors("high", axis='x')
    _yerrlo, _yerrhi = tgasym.errors("low", axis='y'), tgasym.errors("high", axis='y')
    _binwidth = _xerrlo + _xerrhi
    _y = _y * _binwidth
    _yerrlo = _yerrlo * _binwidth
    _yerrhi = _yerrhi * _binwidth
    return _x, _y, [_yerrlo, _yerrhi], [_xerrlo, _xerrhi]


# Poisson error calculation (square root of the bin count)
def poisson_errors(counts):
    return np.sqrt(counts)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", default='', help="Model/Fit dir")
parser.add_argument("-i", "--input", default='fitDiagnosticsTest.root', help="Input shapes file")
parser.add_argument("--fit", default='prefit', choices={"prefit", "fit_s"}, dest='fit', help="Shapes to plot")
parser.add_argument("-o", "--output-folder", default='plots', dest='output_folder', help="Folder to store plots - will be created if it doesn't exist.")
parser.add_argument("--year", default=None, choices={"2016", "2017", "2018", "2016APV"}, type=str, help="year label")
parser.add_argument("--sflabels", default="ParT, V", type=str, help="labels")
parser.add_argument('-f', "--format", type=str, default='png', choices={'png', 'pdf'}, help="Plot format")
parser.add_argument("--scale", default=1, type=float, help="Scale value, as used in template generation. (datacard scaling is parsed automatically)")
parser.add_argument("--smear", default=0.5, type=float, help="Smear value, as used in template generation. (datacard scaling is parsed automatically)")

args = parser.parse_args()

# Setup directories and files
if args.output_folder.split("/")[0] != args.dir:
    args.output_folder = os.path.join(args.dir, args.output_folder)

rd = r.TFile.Open(os.path.join(args.dir, args.input))
fd = uproot.open(os.path.join(args.dir, args.input))

with open(os.path.join(args.dir, 'config.json')) as cfg_file:
    cfg = json.load(cfg_file)

if not os.path.exists(os.path.join(args.dir, 'plots')):
    os.mkdir(os.path.join(args.dir, 'plots'))

shapetype = 'shapes_{}'.format(args.fit)
regions = [r.replace(";1", '') for r in fd[shapetype].keys() if "/" not in r]

lumi = {
    2016: 16.81,
    2017: 41.53,
    2018: 59.74,
    "2016APV": 19.52,
}

# Main loop over regions to generate plots
for i, reg in enumerate(regions):
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0)

    ho1 = convTH1(fd[shapetype + '/' + reg + '/qcd'])
    ho2 = convTH1(fd[shapetype + '/' + reg + '/wqq'])
    tgo = covnTGA(fd[shapetype + '/' + reg + '/data'])

    # Poisson errors for data (tgo[1] is data, and tgo[2] is the error)
    data_poisson_errors = poisson_errors(tgo[1])

    # Poisson errors for MC (ho1[0] is MC histogram counts, ho1[2] is the error)
    mc1_poisson_errors = poisson_errors(ho1[0])
    mc2_poisson_errors = poisson_errors(ho2[0])

    # Plot data with Poisson errors
    ax.errorbar(tgo[0], tgo[1], yerr=data_poisson_errors, xerr=tgo[3], fmt='o', color='black', label='Data')

    # Stack MC histograms with Poisson errors
    hep.histplot([ho1[0], ho2[0]], ho1[1], color=['b', 'grey'], stack=True, ax=ax, label=['Unmatched', "Matched"], histtype='fill')

    ax.set_xlim(40, 180)
    ax.set_ylabel('Events', y=1, ha='right')
    ax.legend()

    # Create ratio (Data/MC)
    data = unumpy.uarray(tgo[1], data_poisson_errors)
    mc1 = unumpy.uarray(ho1[0], mc1_poisson_errors)
    mc2 = unumpy.uarray(ho2[0], mc2_poisson_errors)
    
    ratio = data / (mc1 + mc2)

    # Plot the ratio with Poisson errors
    rax.errorbar(tgo[0], unumpy.nominal_values(ratio), unumpy.std_devs(ratio), xerr=tgo[3], fmt='o', color='black', label='Data/MC')
    rax.hlines(1, 40, 180, linestyle='--', color='k', alpha=0.7)
    rax.set_ylim(0.4, 1.6)
    rax.set_ylabel("Data/MC")
    rax.set_xlabel("Jet $m_{SD}$ [GeV]", x=1, ha='right')

    hep.cms.label(ax=ax, data=True, year=args.year, lumi=lumi[args.year])

    #hep.cms.label(ax=ax, data=True, year=args.year, lumi=lumi[int(args.year)])
    # Save figure
    fig.savefig(f'{args.output_folder}/{shapetype}_{reg}.{args.format}', bbox_inches="tight")

