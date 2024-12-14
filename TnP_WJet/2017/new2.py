import ROOT as r
import uproot
import json
import os
import pprint
import numpy as np
import matplotlib
import warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from matplotlib.offsetbox import AnchoredText
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def rebin_data(vals, edges, new_bins):
    """
    Rebin histogram data to match new bin edges.
    """
    new_vals, _ = np.histogram(edges[:-1], bins=new_bins, weights=vals)
    return new_vals

def poisson_errors(data_vals):
    """
    Calculate Poisson errors for data points.
    Handles zero bins by setting lower errors to 0.
    """
    lower_errors = np.sqrt(data_vals)
    upper_errors = np.sqrt(data_vals)

    # Handle zero bins
    lower_errors[data_vals == 0] = 0
    upper_errors[data_vals == 0] = 0

    return lower_errors, upper_errors

def convTH1(TH1, msdbins):
    """
    Handle both histograms and TGraphAsymmErrors.
    Rebin histogram data if necessary.
    """
    if hasattr(TH1, 'values'):  # Check if it's a histogram
        vals = TH1.values()
        edges = TH1.axes[0].edges()
        variances = TH1.variances()

        # Rebin histogram data
        new_vals = rebin_data(vals, edges, msdbins)
        new_edges = msdbins

    elif isinstance(TH1, r.TGraphAsymmErrors):  # Handle TGraphAsymmErrors
        # Extract points and errors
        x, y = np.array([TH1.GetX()[i] for i in range(TH1.GetN())]), np.array([TH1.GetY()[i] for i in range(TH1.GetN())])
        new_vals, _ = np.histogram(x, bins=msdbins, weights=y)  # Rebin to new bins
        new_edges = msdbins

    else:
        raise TypeError("Unsupported object type for convTH1.")

    return new_vals, new_edges



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", default='', help="Model/Fit dir")
parser.add_argument("-i", "--input", default='fitDiagnosticsTest.root', help="Input shapes file")
parser.add_argument("--fit", default='fit_s', choices={"prefit", "fit_s"}, dest='fit', help="Shapes to plot")
parser.add_argument("-o", "--output-folder", default='plots', dest='output_folder', help="Folder to store plots")
parser.add_argument("--year", default=None, choices={"2016", "2017", "2018"}, type=str, help="year label")
parser.add_argument("--sflabels", default="N2+CvB,CvL", type=str, help="labels")
parser.add_argument('-f', "--format", type=str, default='png', choices={'png', 'pdf'}, help="Plot format")
parser.add_argument("--scale", default=1, type=float, help="Scale value")
parser.add_argument("--smear", default=0.5, type=float, help="Smear value")

args = parser.parse_args()
if args.output_folder.split("/")[0] != args.dir:
    args.output_folder = os.path.join(args.dir, args.output_folder)

rd = r.TFile.Open(os.path.join(args.dir, args.input))
fd = uproot.open(os.path.join(args.dir, args.input))
with open(os.path.join(args.dir, 'config.json')) as cfg_file:
    cfg = json.load(cfg_file)

if not os.path.exists(os.path.join(args.dir, 'plots')):
    os.mkdir(os.path.join(args.dir, 'plots'))

plt.style.use([hep.style.ROOT])

# Custom binning
#msdbins = np.linspace(40, 180, 15)

shapetype = 'shapes_{}'.format(args.fit)
regions = [r.replace(";1", '') for r in fd[shapetype].keys() if "/" not in r]

lumi = {
    2016: 36.33,
    2017: 41.53,
    2018: 59.74,
}

for i, reg in enumerate(regions):
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0)
    
    ho1_vals, ho1_edges = convTH1(fd[shapetype + '/' + reg + '/qcd'], msdbins)
    ho2_vals, ho2_edges = convTH1(fd[shapetype + '/' + reg + '/wqq'], msdbins)
    data_vals, data_edges = convTH1(fd[shapetype + '/' + reg + '/data'], msdbins)

    # Calculate Poisson errors for data
    lower_err, upper_err = poisson_errors(data_vals)

    bin_centers = (ho1_edges[:-1] + ho1_edges[1:]) / 2  # Bin centers

    ax.errorbar(bin_centers, data_vals, yerr=[lower_err, upper_err], fmt='o', color='black', label='Data')
    hep.histplot([ho1_vals, ho2_vals], ho1_edges, stack=True, ax=ax, label=['Unmatched', "Matched"], histtype='fill')

    ax.set_xlim(40, 180)
    ax.set_ylabel('Events', y=1, ha='right')
    ax.legend()

    from uncertainties import unumpy
    data = unumpy.uarray(data_vals, [lower_err, upper_err])
    mc1 = unumpy.uarray(ho1_vals, np.sqrt(ho1_vals))
    mc2 = unumpy.uarray(ho2_vals, np.sqrt(ho2_vals))
    ratio = data / (mc1 + mc2)

    rax.errorbar(bin_centers, unumpy.nominal_values(ratio), unumpy.std_devs(ratio), fmt='o', color='black', label='Data')
    rax.hlines(1, 40, 180, linestyle='--', color='k', alpha=0.7)
    rax.set_ylim(0.4, 1.6)
    rax.set_ylabel("Data/MC")
    rax.set_xlabel("Jet $m_{SD}$ [GeV]", x=1, ha='right')

    hep.cms.label(ax=ax, data=True, year=args.year, lumi=lumi[int(args.year)])
    
    fig.savefig('{}/{}.{}'.format(args.output_folder, shapetype + '_' + reg, 'pdf'), bbox_inches="tight")
    fig.savefig('{}/{}.{}'.format(args.output_folder, shapetype + '_' + reg, 'png'), bbox_inches="tight")

