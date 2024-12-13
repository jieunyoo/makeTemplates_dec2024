"""
Builds hist.Hist templates after adding systematics for all samples.

Author: Farouk Mokhtar
"""

import argparse
import glob
import json
import logging
import os
import pickle as pkl
import warnings

import hist as hist2
import numpy as np
import pandas as pd
import pyarrow
import yaml
from systematics import get_systematic_dict, sigs
from utils import get_common_sample_name, get_finetuned_score, get_xsecweight

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)

sigs = ["ggF", "VBF", "WH", "ZH", "ttH"]
#bkgs = ["TTbar", "WJetsLNu", "SingleTop", "DYJets", "WZQQ", "Diboson", "EWKvjets", "Fake"]
bkgs = ["TTbar", "WJetsLNu", "SingleTop", "WZQQ", "Diboson", "EWKvjets", "DYJets"]


def get_templates(years, channels, samples, samples_dir, regions_sel, model_path, add_fake=False):
    """
    Postprocesses the parquets by applying preselections, and fills templates for different regions.

    Args
        years [list]: years to postprocess (e.g. ["2016APV", "2016"])
        ch [list]: channels to postprocess (e.g. ["ele", "mu"])
        samples [list]: samples to postprocess (e.g. ["ggF", "TTbar", "Data"])
        samples_dir [dict]: points to the path of the parquets for each region
        regions_sel [dict]: key is the name of the region; value is the selection (e.g. `{"pass": (THWW>0.90)}`)
        model_path [str]: path to the ParT finetuned model.onnx
        add_fake [Bool]: if True will include Fake as an additional sample in the output hists

    Returns
        a dict() object hists[region] that contains histograms with 4 axes (Sample, Systematic, Region, mass_observable)

    """

    
    # add extra selections to preselection
    presel = {
        "mu": {
            "fj_mass": "fj_mass>40",
        },
        "ele": {
            "fj_mass": "fj_mass>40",
        },
    }

    mass_binning = [40,70,100,130,180]
    #mass_binning = [40,60,80,100,120,180]
    #mass_binning = [40,180]

    hists = hist2.Hist(
        hist2.axis.StrCategory([], name="Sample", growth=True),
        hist2.axis.StrCategory([], name="Systematic", growth=True),
        hist2.axis.StrCategory([], name="Region", growth=True),
        hist2.axis.Variable(
            #list(range(55, 255, mass_binning)),
            mass_binning,
            name="mass_observable",
            label=r"V reconstructed mass [GeV]",
            overflow=True,
        ),
        storage=hist2.storage.Weight(),
    )

    SYST_DICT = get_systematic_dict(years)

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")

            with open("../fileset/luminosity.json") as f:
                luminosity = json.load(f)[ch][year]

            for sample in os.listdir(samples_dir[year]):

                sample_to_use = get_common_sample_name(sample)

                if sample_to_use not in samples:
                    continue

                is_data = True if sample_to_use == "Data" else False

                logging.info(f"Finding {sample} samples and should combine them under {sample_to_use}")

                out_files = f"{samples_dir[year]}/{sample}/outfiles/"
                parquet_files = glob.glob(f"{out_files}/*_{ch}.parquet")
                pkl_files = glob.glob(f"{out_files}/*.pkl")

                if not parquet_files:
                    logging.info(f"No parquet file for {sample}")
                    continue

                try:
                    data = pd.read_parquet(parquet_files)
                except pyarrow.lib.ArrowInvalid:  # empty parquet because no event passed selection
                    continue

                if len(data) == 0:
                    continue

                # use hidNeurons to get the finetuned scores
                data["THWW"] = get_finetuned_score(data, model_path)

                # drop hidNeurons which are not needed anymore
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                # apply selection
                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

                    if not is_data:
                        if year == '2018':
                            data['temp_JESdown'] = data['met_pt_JES_down']
                            data['temp_JESup'] = data['met_pt_JES_up']
                            data[['met_pt_JES_down', 'met_pt_JES_up']]= data[['temp_JESup', 'temp_JESdown']]

                            data['temp_JERdown'] = data['met_pt_JER_down']
                            data['temp_JERup'] = data['met_pt_JER_up']
                            data[['met_pt_JER_down', 'met_pt_JER_up']]= data[['temp_JERup', 'temp_JERdown']]
                            
                            data['temp_JES_FlavorQCD_down'] = data['met_pt_JES_FlavorQCD_down']
                            data['temp_JES_FlavorQCD_up'] = data['met_pt_JES_FlavorQCD_up']
                            data[['met_pt_JES_FlavorQCD_down', 'met_pt_JES_FlavorQCD_up']]= data[['temp_JES_FlavorQCD_up', 'temp_JES_FlavorQCD_down']]

                            data['temp_JESRelativeBal_down'] = data['met_pt_JES_RelativeBal_down']
                            data['temp_JESRelativeBal_up'] = data['met_pt_JES_RelativeBal_up']
                            data[['met_pt_JES_RelativeBal_down', 'met_pt_JES_RelativeBal_up']]= data[['temp_JESRelativeBal_up', 'temp_JESRelativeBal_down']]

                            data['temp_JES_HF_down'] = data['met_pt_JES_HF_down']
                            data['temp_JES_HF_up'] = data['met_pt_JES_HF_up']
                            data[['met_pt_JES_HF_down', 'met_pt_JES_HF_up']]= data[['temp_JES_HF_up', 'temp_JES_HF_down']]
                            
                            data['temp_JES_BBEC1_down'] = data['met_pt_JES_BBEC1_down']
                            data['temp_JES_BBEC1_up'] = data['met_pt_JES_BBEC1_up']
                            data[['met_pt_JES_BBEC1_down', 'met_pt_JES_BBEC1_up']]= data[['temp_JES_BBEC1_up', 'temp_JES_BBEC1_down']]

                            data['temp_JES_EC2_down'] = data['met_pt_JES_EC2_down']
                            data['temp_JES_EC2_up'] = data['met_pt_JES_EC2_up']
                            data[['met_pt_JES_EC2_down', 'met_pt_JES_EC2_up']]= data[['temp_JES_EC2_up', 'temp_JES_EC2_down']]

                            data['temp_JES_Absolute_down'] = data['met_pt_JES_Absolute_down']
                            data['temp_JES_Absolute_up'] = data['met_pt_JES_Absolute_up']
                            data[['met_pt_JES_Absolute_down', 'met_pt_JES_Absolute_up']]= data[['temp_JES_Absolute_up', 'temp_JES_Absolute_down']]
                            
                            data['temp_JES_BBEC1_2018_down'] = data['met_pt_JES_BBEC1_2018_down']
                            data['temp_JES_BBEC1_2018_up'] = data['met_pt_JES_BBEC1_2018_up']
                            data[['met_pt_JES_BBEC1_2018_down', 'met_pt_JES_BBEC1_2018_up']]= data[['temp_JES_BBEC1_2018_up', 'temp_JES_BBEC1_2018_down']]

                            data['temp_JES_RelativeSample_2018_down'] = data['met_pt_JES_RelativeSample_2018_down']
                            data['temp_JES_RelativeSample_2018_up'] = data['met_pt_JES_RelativeSample_2018_up']
                            data[['met_pt_JES_RelativeSample_2018_down', 'met_pt_JES_RelativeSample_2018_up']]= data[['temp_JES_RelativeSample_2018_up', 'temp_JES_RelativeSample_2018_down']]

                            data['temp_JES_EC2_2018_down'] = data['met_pt_JES_EC2_2018_down']
                            data['temp_JES_EC2_2018_up'] = data['met_pt_JES_EC2_2018_up']
                            data[['met_pt_JES_EC2_2018_down', 'met_pt_JES_EC2_2018_up']]= data[['temp_JES_EC2_2018_up', 'temp_JES_EC2_2018_down']]
                            
                            data['temp_JES_HF_2018_down'] = data['met_pt_JES_HF_2018_down']
                            data['temp_JES_HF_2018_up'] = data['met_pt_JES_HF_2018_up']
                            data[['met_pt_JES_HF_2018_down', 'met_pt_JES_HF_2018_up']]= data[['temp_JES_HF_2018_up', 'temp_JES_HF_2018_down']]

                            data['temp_JES_Absolute_2018_down'] = data['met_pt_JES_Absolute_2018_down']
                            data['temp_JES_Absolute_2018_up'] = data['met_pt_JES_Absolute_2018_up']
                            data[['met_pt_JES_Absolute_2018_down', 'met_pt_JES_Absolute_2018_up']]= data[['temp_JES_Absolute_2018_up', 'temp_JES_Absolute_2018_down']]
                            data['temp_JES_Total_down'] = data['met_pt_JES_Total_down']
                            data['temp_JES_Total_up'] = data['met_pt_JES_Total_up']
                            data[['met_pt_JES_Total_down', 'met_pt_JES_Total_up']]= data[['temp_JES_Total_up', 'temp_JES_Total_down']]

                        elif year == '2017':
                            data['temp_JESdown'] = data['met_pt_JES_down']
                            data['temp_JESup'] = data['met_pt_JES_up']
                            data[['met_pt_JES_down', 'met_pt_JES_up']]= data[['temp_JESup', 'temp_JESdown']]
                            data['temp_JERdown'] = data['met_pt_JER_down']
                            data['temp_JERup'] = data['met_pt_JER_up']
                            data[['met_pt_JER_down', 'met_pt_JER_up']]= data[['temp_JERup', 'temp_JERdown']]
        
                            data['temp_JES_FlavorQCD_down'] = data['met_pt_JES_FlavorQCD_down']
                            data['temp_JES_FlavorQCD_up'] = data['met_pt_JES_FlavorQCD_up']
                            data[['met_pt_JES_FlavorQCD_down', 'met_pt_JES_FlavorQCD_up']]= data[['temp_JES_FlavorQCD_up', 'temp_JES_FlavorQCD_down']]

                            data['temp_JESRelativeBal_down'] = data['met_pt_JES_RelativeBal_down']
                            data['temp_JESRelativeBal_up'] = data['met_pt_JES_RelativeBal_up']
                            data[['met_pt_JES_RelativeBal_down', 'met_pt_JES_RelativeBal_up']]= data[['temp_JESRelativeBal_up', 'temp_JESRelativeBal_down']]

                            data['temp_JES_HF_down'] = data['met_pt_JES_HF_down']
                            data['temp_JES_HF_up'] = data['met_pt_JES_HF_up']
                            data[['met_pt_JES_HF_down', 'met_pt_JES_HF_up']]= data[['temp_JES_HF_up', 'temp_JES_HF_down']]
                            
                            data['temp_JES_BBEC1_down'] = data['met_pt_JES_BBEC1_down']
                            data['temp_JES_BBEC1_up'] = data['met_pt_JES_BBEC1_up']
                            data[['met_pt_JES_BBEC1_down', 'met_pt_JES_BBEC1_up']]= data[['temp_JES_BBEC1_up', 'temp_JES_BBEC1_down']]

                            data['temp_JES_EC2_down'] = data['met_pt_JES_EC2_down']
                            data['temp_JES_EC2_up'] = data['met_pt_JES_EC2_up']
                            data[['met_pt_JES_EC2_down', 'met_pt_JES_EC2_up']]= data[['temp_JES_EC2_up', 'temp_JES_EC2_down']]

                            data['temp_JES_Absolute_down'] = data['met_pt_JES_Absolute_down']
                            data['temp_JES_Absolute_up'] = data['met_pt_JES_Absolute_up']
                            data[['met_pt_JES_Absolute_down', 'met_pt_JES_Absolute_up']]= data[['temp_JES_Absolute_up', 'temp_JES_Absolute_down']]
                            
                            data['temp_JES_BBEC1_2017_down'] = data['met_pt_JES_BBEC1_2017_down']
                            data['temp_JES_BBEC1_2017_up'] = data['met_pt_JES_BBEC1_2017_up']
                            data[['met_pt_JES_BBEC1_2017_down', 'met_pt_JES_BBEC1_2017_up']]= data[['temp_JES_BBEC1_2017_up', 'temp_JES_BBEC1_2017_down']]

                            data['temp_JES_RelativeSample_2017_down'] = data['met_pt_JES_RelativeSample_2017_down']
                            data['temp_JES_RelativeSample_2017_up'] = data['met_pt_JES_RelativeSample_2017_up']
                            data[['met_pt_JES_RelativeSample_2017_down', 'met_pt_JES_RelativeSample_2017_up']]= data[['temp_JES_RelativeSample_2017_up', 'temp_JES_RelativeSample_2017_down']]

                            data['temp_JES_EC2_2017_down'] = data['met_pt_JES_EC2_2017_down']
                            data['temp_JES_EC2_2017_up'] = data['met_pt_JES_EC2_2017_up']
                            data[['met_pt_JES_EC2_2017_down', 'met_pt_JES_EC2_2017_up']]= data[['temp_JES_EC2_2017_up', 'temp_JES_EC2_2017_down']]
                            
                            data['temp_JES_HF_2017_down'] = data['met_pt_JES_HF_2017_down']
                            data['temp_JES_HF_2017_up'] = data['met_pt_JES_HF_2017_up']
                            data[['met_pt_JES_HF_2017_down', 'met_pt_JES_HF_2017_up']]= data[['temp_JES_HF_2017_up', 'temp_JES_HF_2017_down']]

                            data['temp_JES_Absolute_2017_down'] = data['met_pt_JES_Absolute_2017_down']
                            data['temp_JES_Absolute_2017_up'] = data['met_pt_JES_Absolute_2017_up']
                            data[['met_pt_JES_Absolute_2017_down', 'met_pt_JES_Absolute_2017_up']]= data[['temp_JES_Absolute_2017_up', 'temp_JES_Absolute_2017_down']]
                            data['temp_JES_Total_down'] = data['met_pt_JES_Total_down']
                            data['temp_JES_Total_up'] = data['met_pt_JES_Total_up']
                            data[['met_pt_JES_Total_down', 'met_pt_JES_Total_up']]= data[['temp_JES_Total_up', 'temp_JES_Total_down']]

                # get the xsecweight
                xsecweight, sumgenweights, sumpdfweights, sumscaleweights = get_xsecweight(
                    pkl_files, year, sample, sample_to_use, is_data, luminosity
                )

                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                    df = data.copy()

                    #logging.info(f"Applying {region} selection on {len(df)} events")
                    df = df.query(region_sel)
                    #logging.info(f"Will fill the histograms with the remaining {len(df)} events")

                    # ------------------- Nominal -------------------
                    if is_data:
                        nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                    else:
                        nominal = df[f"weight_{ch}"] * xsecweight

                        if "numberBJets" in region_sel:  # if there's a bjet selection, add btag SF to the nominal weight
                            nominal *= df["weight_btag"]

                        if sample_to_use == "TTbar":
                            nominal *= df["top_reweighting"]

                        if sample_to_use == "TTbar" or sample_to_use == "WH" or sample_to_use == "ZH" or sample_to_use == "ggF" or sample_to_use == "VBF" or sample_to_use == "ttH" or sample_to_use == "SingleTop" or sample_to_use == "WZQQ" or sample_to_use == "Diboson":
                            nominal *= 0.8

                    ###################################
                    if sample_to_use == "EWKvjets":
                        threshold = 20
                        df = df[nominal < threshold]
                        nominal = nominal[nominal < threshold]
                    ###################################

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="nominal",
                        Region=region,
                        mass_observable=df["fj_mass"],
                        weight=nominal,
                    )

                    # ------------------- PDF acceptance -------------------

                    """
                    For the PDF acceptance uncertainty:
                    - store 103 variations. 0-100 PDF values
                    - The last two values: alpha_s variations.
                    - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
                    e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
                    and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488
                    """
                    # if sample_to_use in sigs:
                    if (sample_to_use in sigs + ["WJetsLNu", "TTbar"]) and (sample != "ST_s-channel_4f_hadronicDecays"):

                        pdfweights = []

                        for weight_i in sumpdfweights:

                            # noqa: get the normalization factor per variation i (ratio of sumpdfweights_i/sumgenweights)
                            R_i = sumpdfweights[weight_i] / sumgenweights

                            pdfweight = df[f"weight_pdf{weight_i}"].values * nominal / R_i
                            pdfweights.append(pdfweight)

                        pdfweights = np.swapaxes(np.array(pdfweights), 0, 1)  # so that the shape is (# events, variation)

                        abs_unc = np.linalg.norm((pdfweights - nominal.values.reshape(-1, 1)), axis=1)
                        # cap at 100% uncertainty
                        rel_unc = np.clip(abs_unc / nominal, 0, 1)
                        shape_up = nominal * (1 + rel_unc)
                        shape_down = nominal * (1 - rel_unc)

                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_pdf_acceptance_up",
                        Region=region,
                        mass_observable=df["fj_mass"],
                        weight=shape_up,
                    )

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_pdf_acceptance_down",
                        Region=region,
                        mass_observable=df["fj_mass"],
                        weight=shape_down,
                    )

                    # ------------------- QCD scale -------------------

                    """
                    For the QCD acceptance uncertainty:
                    - we save the individual weights [0, 1, 3, 5, 7, 8]
                    - postprocessing: we obtain sum_sumlheweight
                    - postprocessing: we obtain LHEScaleSumw: sum_sumlheweight[i] / sum_sumgenweight
                    - postprocessing:
                    obtain histograms for 0, 1, 3, 5, 7, 8 and 4: h0, h1, ... respectively
                    weighted by scale_0, scale_1, etc
                    and normalize them by  (xsec * luminosity) / LHEScaleSumw[i]
                    - then, take max/min of h0, h1, h3, h5, h7, h8 w.r.t h4: h_up and h_dn
                    - the uncertainty is the nominal histogram * h_up / h4
                    """
                    if (sample_to_use in sigs + ["WJetsLNu", "TTbar", "SingleTop"]) and (
                        sample != "ST_s-channel_4f_hadronicDecays"
                    ):

                        R_4 = sumscaleweights[4] / sumgenweights
                        scaleweight_4 = df["weight_scale4"].values * nominal / R_4

                        scaleweights = []
                        for weight_i in sumscaleweights:
                            if weight_i == 4:
                                continue

                            # get the normalization factor per variation i (ratio of sumscaleweights_i/sumgenweights)
                            R_i = sumscaleweights[weight_i] / sumgenweights
                            scaleweight_i = df[f"weight_scale{weight_i}"].values * nominal / R_i

                            scaleweights.append(scaleweight_i)

                        scaleweights = np.array(scaleweights)

                        scaleweights = np.swapaxes(
                            np.array(scaleweights), 0, 1
                        )  # so that the shape is (# events, variation)

                        # TODO: debug
                        shape_up = nominal * np.max(scaleweights, axis=1) / scaleweight_4
                        shape_down = nominal * np.min(scaleweights, axis=1) / scaleweight_4

                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_qcd_scale_up",
                        Region=region,
                        mass_observable=df["fj_mass"],
                        weight=shape_up,
                    )

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="weight_qcd_scale_down",
                        Region=region,
                        mass_observable=df["fj_mass"],
                        weight=shape_down,
                    )

                    # ------------------- Top pt reweighting systematic  -------------------

                    if sample_to_use == "TTbar":
                        # first remove the reweighting effect
                        nominal_noreweighting = nominal / df["top_reweighting"]

                        shape_up = nominal_noreweighting * (df["top_reweighting"] ** 2)  # "up" is twice the correction
                        shape_down = nominal_noreweighting  # "down" is no correction
                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="top_reweighting_up",
                        Region=region,
                        mass_observable=df["fj_mass"],
                        weight=shape_up,
                    )

                    hists.fill(
                        Sample=sample_to_use,
                        Systematic="top_reweighting_down",
                        Region=region,
                        mass_observable=df["fj_mass"],
                        weight=shape_down,
                    )

                    # ------------------- Common systematics  -------------------

                    for syst, (yrs, smpls, var) in SYST_DICT["common"].items():
                        #print('syst', syst)

                        if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                            shape_up = df[var[ch] + "Up"] * xsecweight
                            shape_down = df[var[ch] + "Down"] * xsecweight

                            if "numberBJets" in region_sel:  # if there's a bjet selection, add btag SF to the nominal weight
                                shape_up *= df["weight_btag"]
                                shape_down *= df["weight_btag"]

                            if sample_to_use == "TTbar":
                                shape_up *= df["top_reweighting"]
                                shape_down *= df["top_reweighting"]
                        else:
                            shape_up = nominal
                            shape_down = nominal

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_up",
                            Region=region,
                            mass_observable=df["fj_mass"],
                            weight=shape_up,
                        )

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_down",
                            Region=region,
                            mass_observable=df["fj_mass"],
                            weight=shape_down,
                        )

                    # ------------------- btag systematics  -------------------

                    for syst, (yrs, smpls, var) in SYST_DICT["btag"].items():

                        if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                            shape_up = df[var[ch] + "Up"] * nominal
                            shape_down = df[var[ch] + "Down"] * nominal
                        else:
                            shape_up = nominal
                            shape_down = nominal

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_up",
                            Region=region,
                            mass_observable=df["fj_mass"],
                            weight=shape_up,
                        )

                        hists.fill(
                            Sample=sample_to_use,
                            Systematic=f"{syst}_down",
                            Region=region,
                            mass_observable=df["fj_mass"],
                            weight=shape_down,
                        )


                    for syst, (yrs, smpls, var) in SYST_DICT["JEC_systs_MASS"].items():
                        if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                            #print('got it')
                            shape_up = df["fj_mass" + var[ch] + "_up"]
                            shape_down = df["fj_mass" + var[ch] + "_down"]
                        else:
                            shape_up = df["fj_mass"]
                            shape_down = df["fj_mass"]
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=shape_up, weight=nominal, )
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=shape_down, weight=nominal, )

                # ------------------- individual sources of JES -------------------

                """We apply the jet pt cut on the up/down variations. Must loop over systematics first."""
                for syst, (yrs, smpls, var) in SYST_DICT["JEC"].items():
                    for variation in ["up", "down"]:
                        for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.

                            if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                #region_sel = region_sel.replace("fj_pt", "fj_pt" + var[ch] + f"_{variation}")

                                region_sel = region_sel.replace("fj_pt", "fj_pt" + var[ch] + f"_{variation}")
                                #region_sel = region_sel.replace("met", "met" + var[ch] + f"_{variation}")

                                region_sel = region_sel.replace("met_pt", "met_pt_" + var[ch] + f"_{variation}")
                                #print('region_sel', region_sel)


                            df = data.copy()
                            df = df.query(region_sel)

                            # ------------------- Nominal -------------------
                            if is_data:
                                nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                            else:
                                nominal = df[f"weight_{ch}"] * xsecweight

                                if "numberBJets" in region_sel:  # if there's a bjet selection, add btag SF to the nominal weight
                                    nominal *= df["weight_btag"]

                                if sample_to_use == "TTbar":
                                    nominal *= df["top_reweighting"]

                            ###################################
                            if sample_to_use == "EWKvjets":
                                threshold = 20
                                df = df[nominal < threshold]
                                nominal = nominal[nominal < threshold]
                            ###################################

                            if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                shape_variation = df["fj_mass"] # + var[ch] + f"_{variation}"]
                            else:
                                shape_variation = df["fj_mass"]

                            hists.fill(
                                Sample=sample_to_use,
                                Systematic=f"{syst}_{variation}",
                                Region=region,
                                mass_observable=shape_variation,
                                weight=nominal,
                            )

 #   for variation in ["fakes_nominal", "fakes_SF_Up", "fakes_SF_Down"]:
 #       for year in years:
        #data = pd.read_parquet(f"{samples_dir[year]}/fake_{year}_ele.parquet")
            #data = pd.read_parquet(f"/uscms/home/jieun201/nobackup/YOURWORKINGAREA/Fake_{year}/outfiles/0-1_ele.parquet")
#            data = pd.read_parquet(f"/uscms/home/jieun201/nobackup/YOURWORKINGAREA/Fake_{year}/outfiles/{variation}_ele.parquet")
            #print('data', data)
#            for selection in presel["ele"]:
                #logging.info(f"Applying {selection} selection on {len(data)} events")
#                data = data.query(presel["ele"][selection])
#                data["THWW"] = get_finetuned_score(data, model_path)
#            for region in hists.axes["Region"]:
#                df = data.copy()
#                logging.info(f"Applying {region} selection on {len(data)} events")
#                df = df.query(regions_sel[region])
#                logging.info(f"Will fill the histograms with the remaining {len(data)} events")
               # print('df', df['fj_mass'])
            #need to rename nonprompt_event_weight as event_weight
                #hists.fill( Sample="Fake", Systematic="nominal", Region=region, mass_observable=df["fj_mass"], weight=df["event_weight"],  )
#                if variation == "fakes_nominal":
#                    hists.fill( Sample="Fake", Systematic="nominal", Region=region, mass_observable=df["fj_mass"], weight=df["event_weight"],  )
#                else:
#                    print('variation', variation)
#                    hists.fill( Sample="Fake", Systematic=variation, Region=region, mass_observable=df["fj_mass"], weight=df["event_weight"],  )


    return hists

#NOTE - by default this is off
    if add_fake:
        for variation in ["FR_Nominal", "FR_stat_Up", "FR_stat_Down", "EWK_SF_Up", "EWK_SF_Down"]:
            for year in years:
                data = pd.read_parquet(f"{samples_dir[year]}/fake_{year}_ele_{variation}.parquet")
                # apply selection
                for selection in presel["ele"]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel["ele"][selection])
                data["event_weight"] *= 0.6  # the closure test SF
                for region in hists.axes["Region"]:
                    df = data.copy()
                    logging.info(f"Applying {region} selection on {len(df)} events")
                    df = df.query(regions_sel[region])
                    logging.info(f"Will fill the histograms with the remaining {len(df)} events")

                    if variation == "FR_Nominal":
                        hists.fill(
                            Sample="Fake",
                            Systematic="nominal",
                            Region=region,
                            mass_observable=df["fj_mass"],
                            weight=df["event_weight"],
                        )
                    else:
                        hists.fill(
                            Sample="Fake",
                            Systematic=variation,
                            Region=region,
                            mass_observable=df["fj_mass"],
                            weight=df["event_weight"],
                        )

    logging.info(hists)

    return hists


def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            neg_bins = np.where(h[{"Sample": sample, "Systematic": "nominal", "Region": region}].values() < 0)[0]

            if len(neg_bins) > 0:
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                region_index = np.argmax(np.array(h.axes["Region"]) == region)

                for neg_bin in neg_bins:
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 1e-3
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 1e-3


def main(args):
    years = args.years.split(",")
    channels = args.channels.split(",")
    with open("config_make_templates.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if len(years) == 4:
        save_as = "Run2"
    else:
        save_as = "_".join(years)

    if len(channels) == 1:
        save_as += f"_{channels[0]}_"

    os.system(f"mkdir -p {args.outdir}")

    hists = get_templates(
        years,
        channels,
        config["samples"],
        config["samples_dir"],
        config["regions_sel"],
        config["model_path"],
        args.add_fake,
    )

    fix_neg_yields(hists)

    with open(f"{args.outdir}/hists_templates_{save_as}.pkl", "wb") as fp:
        pkl.dump(hists, fp)


if __name__ == "__main__":
    # e.g.
    # python make_templates.py --years 2016,2016APV,2017,2018 --channels mu,ele --outdir templates/v1 --add-fake

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", dest="years", default="2017", help="years separated by commas")
    parser.add_argument("--channels", dest="channels", default="mu", help="channels separated by commas (e.g. mu,ele)")
    parser.add_argument("--outdir", dest="outdir", default="templates/test", type=str, help="path of the output")
    parser.add_argument("--add-fake", dest="add_fake", action="store_true")

    args = parser.parse_args()

    main(args)
