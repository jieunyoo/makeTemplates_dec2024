from __future__ import print_function, division
import warnings
import rhalphalib as rl
import pickle
import numpy as np
np.set_printoptions(linewidth=1000, precision=2)
import ROOT
import uproot
from copy import deepcopy
from rhalphalib import AffineMorphTemplate, MorphHistW2
import math
#from util import make_dirs
rl.util.install_roofit_helpers()

SF = {
    "2016APV": {
        'V_SF': 0.927,
        'V_SF_ERR': 0.027,
        'shift_SF': 0.482,
        'shift_SF_ERR': 0.264,
        'smear_SF': 1.156,
        'smear_SF_ERR': 0.023,
    },
 
}



def shape_to_num2(var, nom, clip=1.5):
    nom_rate = np.sum(nom)
    var_rate = np.sum(var)
    if abs(var_rate/nom_rate) > clip:
        var_rate = clip*nom_rate
    if var_rate < 0:
        var_rate = 0
    return var_rate/nom_rate

def get_template2(f, region, sample, syst=None, muon=False, nowarn=False):    
    hist_name = '{}_{}'.format(sample, region)
    #print('hist_name', hist_name)
    if syst is not None:
        hist_name += "_" + syst
    else:
        hist_name += "_nominal" 
    h_vals = f[hist_name].values()
    h_edges = f[hist_name].axes[0].edges()
    h_variances = f[hist_name].variances()
    if np.any(h_vals < 0):
        print("Sample {}, {}, {}, {}, has {} negative bins. They will be set to 0.".format(
            sample, region,syst, np.sum(h_vals < 0)))
        _invalid = h_vals < 0
        h_vals[_invalid] = 0
        h_variances[_invalid] = 0
    if np.any(~np.isfinite(h_vals)):
        print("Sample {}, {}, {}, {}, has {} Nan/Inf bins. They will be set to 0.".format(
            sample, region, syst, np.sum(~np.isfinite(h_vals))))
        _invalid = ~np.isfinite(h_vals)
        h_vals[_invalid] = 0
        h_variances[_invalid] = 0
    return (h_vals, h_variances)


def get_bins():
    #ptbins = np.array([40,60,80,100,120,140,180])
    ptbins = np.array([40,70,110,140,180])
    return ptbins
    
def smass(sName):
    if sName in ['ggF','VBF','WH','ZH','ttH']:
        _mass = 125.
    elif sName in ['WJetsLNu','EWKvjets','TTbar','SingleTop','Diboson', 'WZQQ']:
        _mass = 80.379
    elif sName in ['DYJets','Zjetsbb','EWKZ','EWKZbb']:
        _mass = 91.
    else:
        raise ValueError("What is {}".format(sName))
    return _mass

def badtemp(hvalues, eps=0.0000001, mask=None):
    # Need minimum size & more than 1 non-zero bins
    tot = np.sum(hvalues[mask])
    count_nonzeros = np.sum(hvalues[mask] > 0)
    if (tot < eps) or (count_nonzeros < 2):
        return True
    else:
        return False

def shape_to_num(f, region, sName, syst,muon=False, bound=0.5):
    _nom = get_templ(f, region, sName, muon=muon)
    #print('_nom',_nom)
    if _nom is None:
        return None
    _nom_rate = np.sum(_nom[0])
    if _nom_rate < .1:
        return 1.0

    _one_side = get_templ(f, region, sName, syst=syst, muon=muon, nowarn=True)
    _up = get_templ(f, region, sName, syst=syst, muon=muon, nowarn=True)
    _down = get_templ(f, region, sName, syst=syst, muon=muon, nowarn=True)
    #_up = get_templ(f, region, sName, syst=syst + "Up", muon=muon, nowarn=True)
    #_down = get_templ(f, region, sName, syst=syst + "Down", muon=muon, nowarn=True)

    if _up is None and _down is None and _one_side is None:
        print('got none')
        return None
    else:
        if _one_side is not None:
            _up_rate = np.sum(_one_side[0] )
            _diff = np.abs(_up_rate - _nom_rate)
            magnitude = _diff / _nom_rate
        elif _down is not None and _up is not None:
            _up_rate = np.sum(_up[0] )
            _down_rate = np.sum(_down[0] )
            _diff = np.abs(_up_rate - _nom_rate) + np.abs(_down_rate - _nom_rate)
            magnitude = _diff / (2. * _nom_rate)
        else:
            raise NotImplementedError
    if bound is not None:
        magnitude = min(magnitude, bound)
    return 1.0 + magnitude

def get_templ(f, region, sample, syst=None, muon=False, nowarn=False):
    #print('f', f)    
    if "16" in f:
        year = 2016
    elif "17" in f:
        year = 2017
    else:
        year = 2018
        
    hist_name = '{}_{}'.format(sample, region)
    #print('hist_name', hist_name)

    if syst is not None:
        hist_name += "_" + syst
    else:
        hist_name += "_nominal"
        
    try:
        f[hist_name]
    except:
        if syst is not None and nowarn == False:
            if "HEM" in syst and year in [2016, 2017]:  # always empty
                pass
            elif "HEM" in syst and ("Up" in syst or "Down" in syst) and year == 2018:  # always empty
                pass
            elif "L1Prefiring" in syst and year == 2018:  # always empty
                pass
            else:
                print("{}Sample {}, {}, {}, {} not found.".format('(Muon) ' if muon else "",
                     sample, region if not muon else "-", syst))
        return None
    h_vals = f[hist_name].values()
    h_edges = f[hist_name].axes[0].edges()
    h_variances = f[hist_name].variances()
    
    if np.any(h_vals < 0):
        print("Sample {}, {}, {}, {}, has {} negative bins. They will be set to 0.".format(
            sample, region,syst, np.sum(h_vals < 0)))
        _invalid = h_vals < 0
        h_vals[_invalid] = 0
        h_variances[_invalid] = 0
    if np.any(~np.isfinite(h_vals)):
        print("Sample {}, {}, {}, {}, has {} Nan/Inf bins. They will be set to 0.".format(
            sample, region, syst, np.sum(~np.isfinite(h_vals))))
        _invalid = ~np.isfinite(h_vals)
        h_vals[_invalid] = 0
        h_variances[_invalid] = 0
    h_key = 'msd'
    return (h_vals, h_edges, h_key, h_variances)

def one_bin(template):
    try:
        h_vals, h_edges, h_key, h_variances = template
        return (np.array([np.sum(h_vals)]), np.array([0., 1.]), "onebin", np.array([np.sum(h_variances)]))
    except:
        h_vals, h_edges, h_key = template
        return (np.array([np.sum(h_vals)]), np.array([0., 1.]), "onebin")

#this is the rhalphabet code for the model****************************************************************************

def dummy_rhalphabet( scale_syst=True, smear_syst=True,  systs=True,muonCR=False,year='2016APV',  opts=None):

    #assert year in ['2016', '2017', '2018']

    # Default lumi (needs at least one systematics for prefit)
    sys_lumi = rl.NuisanceParameter('CMS_lumi_13TeV_{}'.format(year), 'lnN')
    sys_lumi_correlated = rl.NuisanceParameter('CMS_lumi_13TeV_correlated', 'lnN')
    sys_lumi_1718 = rl.NuisanceParameter('CMS_lumi_13TeV_1718', 'lnN')
    lumi_dict = {
        "2016APV": 1.01,
        "2016": 1.01,
        "2017": 1.02,
        "2018": 1.015,
    }
    lumi_correlated_dict = {
        "2016APV": 1.006,
        "2016": 1.006,
        "2017": 1.009,
        "2018": 1.02,
    }
    lumi_1718_dict = {
        "2017": 1.006,
        "2018": 1.002,
    }
    # TT params
    tqqeffSF = rl.IndependentParameter('tqqeffSF_{}'.format(year), 1., 0, 10)
    tqqnormSF = rl.IndependentParameter('tqqnormSF_{}'.format(year), 1., 0, 10)
    

    #*********************************************************************************
    # Systematics
    sys_scale = rl.NuisanceParameter('CMS_HWW_boosted_jms_{}'.format(year), 'shape') #this is "scale"
    sys_smear = rl.NuisanceParameter('CMS_HWW_boosted_jmr_{}'.format(year), 'shape') #this is "smear"
    sys_veff = rl.NuisanceParameter('CMS_HWW_boosted_veff_{}'.format(year), 'lnN')
     
    #systematics that are just set a priori
    sys_qcd_scale_WH = rl.NuisanceParameter('QCDscale_WH', 'lnN')
    sys_qcd_scale_ZH = rl.NuisanceParameter('QCDscale_ZH', 'lnN')
    sys_qcd_scale_ggF = rl.NuisanceParameter('QCDscale_ggF', 'lnN')
    sys_qcd_scale_VBF = rl.NuisanceParameter('QCDscale_VBF', 'lnN')
    sys_qcd_scale_ttH = rl.NuisanceParameter('QCDscale_ttH', 'lnN')

    sys_pdf_Higgs_WH = rl.NuisanceParameter('pdf_Higgs_WH', 'lnN')
    sys_pdf_Higgs_ZH = rl.NuisanceParameter('pdf_Higgs_ZH', 'lnN')
    sys_pdf_Higgs_ttH = rl.NuisanceParameter('pdf_Higgs_ttH', 'lnN')
    sys_pdf_Higgs_qqH = rl.NuisanceParameter('pdf_Higgs_qqH', 'lnN')
    sys_pdf_Higgs_ggH = rl.NuisanceParameter('pdf_Higgs_ggH', 'lnN')

    sys_miniisolation_SF_uncert = rl.NuisanceParameter('CMS_HWW_boosted_miniisolation_SF_unc', 'lnN')

    #only for signal
    sys_br_ww = rl.NuisanceParameter('BR_hww', 'lnN')
    sys_tagger = rl.NuisanceParameter('CMS_HWW_boosted_tagger', 'lnN')
    sys_alpha = rl.NuisanceParameter('alpha_s', 'lnN')


    #only for fakes
    #fake_SF_uncert = rl.NuisanceParameter("CMS_HWW_boosted_Fake_SF_uncertainty", "lnN")


    #***SHAPES*************************************************************************
    sys_shape_dict = {}
    #btagging: 4, applied to everything except data, fakes - convert these shapes to lnN visa shape to num function
    sys_shape_dict['weight_btagSFlightCorrelated'] = rl.NuisanceParameter('CMS_HWW_boosted_btagSFlightCorrelated', 'lnN')
    sys_shape_dict['weight_btagSFbcCorrelated'] = rl.NuisanceParameter('CMS_HWW_boosted_btagSFbcCorrelated', 'lnN')
    sys_shape_dict['weight_btagSFlight'] = rl.NuisanceParameter('CMS_HWW_boosted_btagSFlightCorrelated_{}'.format(year), 'lnN')
    sys_shape_dict['weight_btagSFbc'] = rl.NuisanceParameter('CMS_HWW_boosted_btagSFbcCorrelated_{}'.format(year), 'lnN')

    #these are shapes, convert to lnN
    sys_shape_dict['weight_d1K_NLO'] = rl.NuisanceParameter('CMS_HWW_boosted_d1K_NLO', 'lnN')
    sys_shape_dict['weight_d2K_NLO'] = rl.NuisanceParameter('CMS_HWW_boosted_d2K_NLO', 'lnN') 
    sys_shape_dict['weight_d3K_NLO'] = rl.NuisanceParameter('CMS_HWW_boosted_d3K_NLO', 'lnN')
    sys_shape_dict['weight_d1kappa_EW'] = rl.NuisanceParameter('CMS_HWW_boosted_W_d1kappa_EW', 'lnN')
    sys_shape_dict['weight_W_d2kappa_EW'] = rl.NuisanceParameter('CMS_HWW_boosted_W_d2kappa_EW','lnN')
    sys_shape_dict['weight_W_d3kappa_EW'] = rl.NuisanceParameter('CMS_HWW_boosted_W_d3kappa_EW','lnN')
    sys_shape_dict['weight_Z_d2kappa_EW'] = rl.NuisanceParameter('CMS_HWW_boosted_Z_d2kappa_EW','lnN')
    sys_shape_dict['weight_Z_d3kappa_EW'] = rl.NuisanceParameter('CMS_HWW_boosted_Z_d3kappa_EW', 'lnN')
    sys_shape_dict['weight_mu_isolation'] = rl.NuisanceParameter('CMS_HWW_boosted_mu_isolation', 'lnN')
    sys_shape_dict['weight_mu_trigger_noniso'] = rl.NuisanceParameter('CMS_HWW_boosted_mu_trigger', 'lnN')
    sys_shape_dict['weight_mu_id'] = rl.NuisanceParameter('CMS_HWW_boosted_mu_identification', 'lnN')
    sys_shape_dict['weight_mu_trigger_iso'] = rl.NuisanceParameter('CMS_HWW_boosted_mu_trigger_iso', 'lnN')
    sys_shape_dict['weight_ele_reco'] = rl.NuisanceParameter('CMS_HWW_boosted_ele_reconstruction', 'lnN')
    sys_shape_dict['weight_ele_id'] = rl.NuisanceParameter('CMS_HWW_boosted_ele_identification', 'lnN')
    #******************************

    #general shapes
    sys_shape_dict['weight_pileup'] = rl.NuisanceParameter('CMS_pileup_{}'.format(year), 'lnN')
    sys_shape_dict['weight_pileup_id'] = rl.NuisanceParameter('CMS_pileup_id', 'lnN')
    sys_shape_dict["weight_L1Prefiring"] = rl.NuisanceParameter('CMS_l1_ecal_prefiring_{}'.format(year),'shape')


    #JEC related
    sys_shape_dict['UES'] = rl.NuisanceParameter('unclustered_Energy', 'lnN')
    sys_shape_dict['JES_FlavorQCD'] = rl.NuisanceParameter('CMS_scale_j_FlavQCD', 'lnN')
    sys_shape_dict['JES_RelativeBal'] = rl.NuisanceParameter('CMS_scale_j_RelBal', 'lnN')
    sys_shape_dict['JES_HF'] = rl.NuisanceParameter('CMS_scale_j_HF', 'lnN')
    sys_shape_dict['JES_BBEC1'] = rl.NuisanceParameter('CMS_scale_j_BBEC1', 'lnN')
    sys_shape_dict['JES_EC2'] = rl.NuisanceParameter('CMS_scale_j_EC2', 'lnN')
    sys_shape_dict['JES_Absolute'] = rl.NuisanceParameter('CMS_scale_j_Abs', 'lnN')

    sys_shape_dict["JER_year"] = rl.NuisanceParameter('CMS_res_j_{}'.format(year),'lnN')
    sys_shape_dict["JES_BBEC1_year"] = rl.NuisanceParameter('CMS_scale_j_BBEC1_{}'.format(year),'lnN')
    sys_shape_dict["JES_RelativeSample_year"] = rl.NuisanceParameter('CMS_scale_j_RelSample_{}'.format(year),'lnN')
    sys_shape_dict["JES_EC2_year"] = rl.NuisanceParameter('CMS_scale_j_EC2_{}'.format(year),'lnN')
    sys_shape_dict["JES_HF_year"] = rl.NuisanceParameter('CMS_scale_j_HF_{}'.format(year),'lnN')
    sys_shape_dict["JES_Absolute_year"] = rl.NuisanceParameter('CMS_scale_j_Abs_{}'.format(year),'lnN')
  


    #theory - applying only to specific samples
    sys_shape_dict['weight_pdf_acceptance_WH'] = rl.NuisanceParameter('PDF_WH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_qcd_scale_WH'] = rl.NuisanceParameter('QCDscale_WH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_pdf_acceptance_ZH'] = rl.NuisanceParameter('PDF_ZH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_qcd_scale_ZH'] = rl.NuisanceParameter('QCDscale_ZH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_pdf_acceptance_ggF'] = rl.NuisanceParameter('PDF_ggH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_qcd_scale_ggF'] = rl.NuisanceParameter('QCDscale_ggH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_pdf_acceptance_VBF'] = rl.NuisanceParameter('PDF_qqH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_qcd_scale_VBF'] = rl.NuisanceParameter('QCDscale_qqH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_pdf_acceptance_ttH'] = rl.NuisanceParameter('PDF_ttH_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_qcd_scale_ttH'] = rl.NuisanceParameter('QCDscale_ttH_ACCEPT_CMS_HWW_boosted', 'shape')

    sys_shape_dict['weight_qcd_scale_TTbar'] = rl.NuisanceParameter('QCDscale_ttbar_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_pdf_acceptance_TTbar'] = rl.NuisanceParameter('PDF_ttbar_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_qcd_scale_SingleTop'] = rl.NuisanceParameter('QCDscale_singletop_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_pdf_acceptance_SingleTop'] = rl.NuisanceParameter('PDF_singletop_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_pdf_acceptance_WJetsLNu'] = rl.NuisanceParameter('PDF_wjets_ACCEPT_CMS_HWW_boosted', 'shape')
    sys_shape_dict['weight_qcd_scale_WJetsLNu'] = rl.NuisanceParameter('QCDscale_wjets_ACCEPT_CMS_HWW_boosted', 'shape')


    sys_shape_dict['weight_PSISR_ggF'] = rl.NuisanceParameter('ps_isr_ggH', 'shape')
    sys_shape_dict['weight_PSISR_VBF'] = rl.NuisanceParameter('ps_isr_qqH', 'shape')
    sys_shape_dict['weight_PSISR_WH'] = rl.NuisanceParameter('ps_isr_WH', 'shape')
    sys_shape_dict['weight_PSISR_ZH'] = rl.NuisanceParameter('ps_isr_ZH', 'shape')
    sys_shape_dict['weight_PSISR_WJetsLNu'] = rl.NuisanceParameter('ps_isr_wjets', 'shape')
    sys_shape_dict['weight_PSISR_ttH'] = rl.NuisanceParameter('ps_isr_ttH', 'shape')
    sys_shape_dict['weight_PSISR_TTbar'] = rl.NuisanceParameter('ps_isr_ttbar', 'shape')
    sys_shape_dict['weight_PSISR_SingleTop'] = rl.NuisanceParameter('ps_isr_singletop', 'shape')

    sys_shape_dict['weight_PSFSR_ggF'] = rl.NuisanceParameter('ps_fsr_ggH', 'shape')
    sys_shape_dict['weight_PSFSR_VBF'] = rl.NuisanceParameter('ps_fsr_qqH', 'shape')
    sys_shape_dict['weight_PSFSR_WH'] = rl.NuisanceParameter('ps_fsr_WH', 'shape')
    sys_shape_dict['weight_PSFSR_ZH'] = rl.NuisanceParameter('ps_fsr_ZH', 'shape')
    sys_shape_dict['weight_PSFSR_WJetsLNu'] = rl.NuisanceParameter('ps_fsr_wjets', 'shape')
    sys_shape_dict['weight_PSFSR_ttH'] = rl.NuisanceParameter('ps_fsr_ttH', 'shape')
    sys_shape_dict['weight_PSFSR_TTbar'] = rl.NuisanceParameter('ps_fsr_ttbar', 'shape')
    sys_shape_dict['weight_PSFSR_SingleTop'] = rl.NuisanceParameter('ps_fsr_singletop', 'shape')

    sys_shape_dict['top_reweighting'] = rl.NuisanceParameter('CMS_HWW_boosted_top_reweighting', 'shape')

    #sys_shape_dict['fakes_SF'] = rl.NuisanceParameter('CMS_HWW_FakeRate_EWK_SF_statistical_uncertainty', 'shape')
    #sys_shape_dict['fakes_DR'] = rl.NuisanceParameter('CMS_HWW_FakeRate_EWK_SF_flavor_uncertainty', 'shape')

    #***********************************************************************************************************************
    msdbins = get_bins()
    print('bins', msdbins)
    msd = rl.Observable('msd', msdbins)
    #model = rl.Model("shapes" + year)  # build actual fit model now
    model = rl.Model("shapes" + year + "_TopCR")  # build actual fit model now _NOTE annotated TopCR here


    f = uproot.open(ROOTFILE)  #open file here
    print('f.keys', f.keys())

    for region in ['pass', 'fail']:
        ch = rl.Channel("TopCR{}{}".format(region, year))
        model.addChannel(ch)
        print('channel', ch)
      
        include_samples = [ # to do: add other samples
                'WJetsLNu', 
            'DYJets', 
            'TTbar', 'SingleTop',
            ]
      
      #just some functions/checking, ignore this section******************************
        from functools import partial
        # badtemp_ma = partial(badtemp, mask=mask)
        def badtemp_ma(hvalues, eps=0.0000001, mask=None):
            # Need minimum size & more than 1 non-zero bins
            tot = np.sum(hvalues[mask])
            count_nonzeros = np.sum(hvalues[mask] > 0)
            if (tot < eps) or (count_nonzeros < 3):
                return True
            else:
                return False
        for sName in include_samples: # Remove empty samples
            templ = get_templ(f, region, sName)
            #print('sName,region', sName, region, templ)
            if templ is None:
                print( 'Sample {} in region = {},not found in template file.' .format(sName, region))
                include_samples.remove(sName)
    #******************************

        for sName in include_samples:  #define signal and background samples
            templ = get_templ(f, region, sName)      


            stype = rl.Sample.BACKGROUND #only use bkg it seems
            sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

 
    #***Morphing of templates here, do for all samples (except data, Fakes)
            MORPHNOMINAL = True
            def smorph(templ,sName):      
                if templ is None:
                    return None                  
                if MORPHNOMINAL and sName not in ['QCD']:
                    return MorphHistW2(templ).get(shift=SF[year]['shift_SF']/smass('WZQQ') * smass(sName), smear=SF[year]['smear_SF'] )
                else:
                    return templ
            templ = smorph(templ, sName) 
            sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)
    #**********************************************************************************

            # Systematics
            #####################################################
            if not systs:  # Need at least one
                sample.setParamEffect(sys_lumi, lumi_dict[year])
                  # if sName not in ['tqq', 'stqq']:
            else:
                sample.setParamEffect(sys_lumi, lumi_dict[year])
                sample.setParamEffect(sys_lumi_correlated, lumi_correlated_dict[year])
                if year != '2016' and year != '2016APV':
                        sample.setParamEffect(sys_lumi_1718, lumi_1718_dict[year])

                sample.setParamEffect(sys_miniisolation_SF_uncert, 1.02, 0.98)

                #apply factors to single samples only below **********************************
                if sName in ['WH']: sample.setParamEffect(sys_qcd_scale_WH,1.005,0.993)
                if sName in ['ZH']: sample.setParamEffect(sys_qcd_scale_ZH,1.038,0.97)
                if sName in ['ggF']: sample.setParamEffect(sys_qcd_scale_ggF,1.039, None)
                if sName in ['VBF']: sample.setParamEffect(sys_qcd_scale_VBF,1.004, 0.997)
                if sName in ['ttH']: sample.setParamEffect(sys_qcd_scale_ttH,1.058, 0.908)

                if sName in ['WH']: sample.setParamEffect(sys_pdf_Higgs_WH,1.017,0.993)
                if sName in ['ZH']: sample.setParamEffect(sys_pdf_Higgs_ZH,1.013,0.97)
                if sName in ['ggF']: sample.setParamEffect(sys_pdf_Higgs_ggH,1.019, None)
                if sName in ['VBF']: sample.setParamEffect(sys_pdf_Higgs_qqH,1.021, 0.997)
                if sName in ['ttH']: sample.setParamEffect(sys_pdf_Higgs_ttH,1.03, 0.908)

                if sName in ['ttH']: sample.setParamEffect(sys_alpha,1.020, None)
                if sName in ['WH']: sample.setParamEffect(sys_alpha,1.009, None)
                if sName in ['ZH']: sample.setParamEffect(sys_alpha,1.009, None)
                if sName in ['VBF']: sample.setParamEffect(sys_alpha,1.005, None)
                if sName in ['ggF']: sample.setParamEffect(sys_alpha,1.026, None)

                if sName in ['WH', 'ZH', 'ggF', 'VBF', 'ttH']: sample.setParamEffect(sys_br_ww,1.0153, 0.9848 )
                if sName in ['WH', 'ZH', 'ggF', 'VBF', 'ttH']: sample.setParamEffect(sys_tagger,1.27,None )

                #********************************************************
                #shapes that you need to convert to --> lnN
                sys_names_toLnN = [
               "weight_d1K_NLO", "weight_d2K_NLO", "weight_d3K_NLO",
                "weight_d1kappa_EW",  "weight_W_d2kappa_EW","weight_W_d3kappa_EW",
                 "weight_Z_d2kappa_EW", "weight_Z_d3kappa_EW",
                 'weight_mu_isolation', 'weight_mu_trigger_noniso', 'weight_mu_trigger_iso', 'weight_mu_id',
                  'weight_ele_id', 'weight_ele_reco', 
                ]
                for sys_name in sys_names_toLnN:
                    print('sName',sys_name, sName)
                    if ("d1K_NLO" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d2K_NLO" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d3K_NLO" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d1kappa_EW" in sys_name) and sName not in ['WJetsLNu', 'DYJets']:
                        continue
                    if ("weight_d2kappa_EW" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d3kappa_EW" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_Z_d2kappa_EW" in sys_name) and sName not in ['DYJets']:
                        continue
                    if ("weight_Z_d3kappa_EW" in sys_name) and sName not in ['DYJets']:
                        continue
             
                    if sys_shape_dict[sys_name].combinePrior == "lnN":
                        nominal = get_template2(f, region, sName,syst='nominal') 
                        syst_up = get_template2(f, region, sName, syst=sys_name + "_up")
                        syst_down = get_template2(f, region, sName, syst=sys_name + "_down")
                              
                        eff_up = shape_to_num2(syst_up,nominal)
                        eff_down = shape_to_num2(syst_down,nominal)
                      
                        if math.isclose(eff_up, eff_down, rel_tol=0.02):  # if up and down are the same
                            arr = np.array([eff_up,eff_down])
                            sample.setParamEffect(sys_shape_dict[sys_name], np.mean(arr))    
                        else:
                            sample.setParamEffect(sys_shape_dict[sys_name], max(eff_up, eff_down), min(eff_up, eff_down))
                    else:
                        nominal[nominal == 0] = 1  # to avoid invalid value encountered in true_divide in "syst_up/nominal"
                        sample.setParamEffect(sys_shape_dict[sys_name], (syst_up / nominal), (syst_down / nominal))

                sys_names_toLnN = [
"weight_btagSFlightCorrelated", 'weight_btagSFbcCorrelated', 'weight_btagSFlight', 'weight_btagSFbc',
                'UES', 'JES_FlavorQCD', 'JES_RelativeBal', 'JES_HF', 'JES_BBEC1_year',  "JES_BBEC1", "JES_EC2",'JES_Absolute',
                "JES_RelativeSample_year", "JES_EC2_year", "JES_HF_year", 'JER_year', 'JES_Absolute_year', 
'weight_pileup', 'weight_pileup_id'
                ]
                for sys_name in sys_names_toLnN:
                    print('sName',sys_name, sName)
                    if ("d1K_NLO" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d2K_NLO" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d3K_NLO" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d1kappa_EW" in sys_name) and sName not in ['WJetsLNu', 'DYJets']:
                        continue
                    if ("weight_d2kappa_EW" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_d3kappa_EW" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_Z_d2kappa_EW" in sys_name) and sName not in ['DYJets']:
                        continue
                    if ("weight_Z_d3kappa_EW" in sys_name) and sName not in ['DYJets']:
                        continue
             
                    if sys_shape_dict[sys_name].combinePrior == "lnN":
                        nominal = get_template2(f, region, sName,syst='nominal') 
                        syst_up = get_template2(f, region, sName, syst=sys_name + "_up")
                        syst_down = get_template2(f, region, sName, syst=sys_name + "_down")
                              
                        eff_up = shape_to_num2(syst_up,nominal)
                        eff_down = shape_to_num2(syst_down,nominal)
                      
                        arr = np.array([eff_up,eff_down])
                        sample.setParamEffect(sys_shape_dict[sys_name], np.mean(arr))    
                #********************************************************   
                #note: only need to do smorphing for shape based templates, so above lnN ones don't use smorphed ones
                #PURELY SHAPE SYSTEMATICS - these are applied to everything
                sys_names = [ 
#"weight_pileup", "weight_pileup_id",
                              'weight_L1Prefiring',
               'weight_pdf_acceptance_WH','weight_qcd_scale_WH',  'weight_pdf_acceptance_ZH','weight_qcd_scale_ZH',
                'weight_pdf_acceptance_VBF','weight_qcd_scale_VBF',  'weight_pdf_acceptance_ggF','weight_qcd_scale_ggF',
                'weight_pdf_acceptance_ttH','weight_qcd_scale_ttH', 
                'weight_pdf_acceptance_TTbar','weight_qcd_scale_TTbar',
                'weight_pdf_acceptance_SingleTop','weight_qcd_scale_SingleTop',
                'weight_pdf_acceptance_WJetsLNu', 'weight_qcd_scale_WJetsLNu',
                'top_reweighting',
                'weight_PSISR_ggF', 'weight_PSISR_VBF', 'weight_PSISR_WH', 'weight_PSISR_ZH', 'weight_PSISR_WJetsLNu', 'weight_PSISR_ttH',
                'weight_PSISR_TTbar', 'weight_PSISR_SingleTop',
                'weight_PSFSR_ggF', 'weight_PSFSR_VBF', 'weight_PSFSR_WH', 'weight_PSFSR_ZH', 'weight_PSFSR_WJetsLNu', 'weight_PSFSR_ttH',
                'weight_PSFSR_TTbar', 'weight_PSFSR_SingleTop',
                ]
  
                for sys_name in sys_names:   
                    #print('sys_name, sample', sys_name, sample)
                    #these individual cases are since the specific weight has a different nuisance parameter named based on sample
                    if ("weight_pdf_acceptance_WH" in sys_name) and sName not in ['WH']:
                        continue
                    if ("weight_qcd_scale_WH" in sys_name) and sName not in ['WH']:
                        continue
                    if ("weight_pdf_acceptance_ZH" in sys_name) and sName not in ['ZH']:
                        continue
                    if ("weight_qcd_scale_ZH" in sys_name) and sName not in ['ZH']:
                        continue
                    if ("weight_pdf_acceptance_ggF" in sys_name) and sName not in ['ggF']:
                        continue
                    if ("weight_qcd_scale_ggF" in sys_name) and sName not in ['ggF']:
                        continue
                    if ("weight_pdf_acceptance_VBF" in sys_name) and sName not in ['VBF']:
                        continue
                    if ("weight_qcd_scale_VBF" in sys_name) and sName not in ['VBF']:
                        continue
                    if ("weight_pdf_acceptance_ttH" in sys_name) and sName not in ['ttH']:
                        continue
                    if ("weight_qcd_scale_ttH" in sys_name) and sName not in ['ttH']:
                        continue

                    if ("weight_pdf_acceptance_TTbar" in sys_name) and sName not in ['TTbar']:
                        continue
                    if ("weight_qcd_scale_TTbar" in sys_name) and sName not in ['TTbar']:
                        continue
                    if ("weight_pdf_acceptance_SingleTop" in sys_name) and sName not in ['SingleTop']:
                        continue
                    if ("weight_qcd_scale_SingleTop" in sys_name) and sName not in ['SingleTop']:
                        continue
                    if ("weight_pdf_acceptance_WJetsLNu" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_qcd_scale_WJetsLNu" in sys_name) and sName not in ['WJetsLNu']:
                        continue

                    if ("weight_PSISR_ggF" in sys_name) and sName not in ['ggF']:
                        continue
                    if ("weight_PSISR_VBF" in sys_name) and sName not in ['VBF']:
                        continue
                    if ("weight_PSISR_ZH" in sys_name) and sName not in ['ZH']:
                        continue
                    if ("weight_PSISR_WH" in sys_name) and sName not in ['WH']:
                        continue
                    if ("weight_PSISR_WJetsLNu" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_PSISR_ttH" in sys_name) and sName not in ['ttH']:
                        continue

                    if ("weight_PSISR_TTbar" in sys_name) and sName not in ['TTbar']:
                        continue
                    if ("weight_PSISR_SingleTop" in sys_name) and sName not in ['SingleTop']:
                        continue

                    if ("weight_PSFSR_ggF" in sys_name) and sName not in ['ggF']:
                        continue
                    if ("weight_PSFSR_VBF" in sys_name) and sName not in ['VBF']:
                        continue
                    if ("weight_PSFSR_ZH" in sys_name) and sName not in ['ZH']:
                        continue
                    if ("weight_PSFSR_WH" in sys_name) and sName not in ['WH']:
                        continue
                    if ("weight_PSFSR_WJetsLNu" in sys_name) and sName not in ['WJetsLNu']:
                        continue
                    if ("weight_PSFSR_ttH" in sys_name) and sName not in ['ttH']:
                        continue

                    if ("weight_PSFSR_TTbar" in sys_name) and sName not in ['TTbar']:
                        continue
                    if ("weight_PSFSR_SingleTop" in sys_name) and sName not in ['SingleTop']:
                        continue

                    if ("top_reweighting" in sys_name) and sName not in ['TTbar']:
                        continue

                    #now get the shapes
                    _up = get_templ(f, region, sName, syst=sys_name + "_up")
                    _dn = get_templ(f, region, sName, syst=sys_name + "_down")
                    _up = smorph(_up, sName)
                    _dn = smorph(_dn, sName)
                    
                    if _up is None or _dn is None:
                        continue
                    if badtemp_ma(_up[0]) or badtemp_ma(_dn[0]):
                        print("Skipping {} for sample {}, shape would be empty".format(sys_name, sName))
                        continue
                    sample.setParamEffect(sys_shape_dict[sys_name],_up[:-1], _dn[:-1])


                if sName not in ["qcd", 'WJetsLNu', 'DYJets',]:
                    sample.scale(SF[year]['V_SF'])
                    sample.setParamEffect(
                        sys_veff, 1.0 + SF[year]['V_SF_ERR'] / SF[year]['V_SF'])


            ################related to smear/scale****************
            # Scale and Smear
            mtempl = AffineMorphTemplate(templ)
            _extra_scaling = 4.
            # if scale_syst and sName not in ['qcd', 'zll', 'wln', 'vvqq', 'tqq', 'stqq']:
            if scale_syst and sName not in ['qcd']:
                realshift = SF[year]['shift_SF_ERR']/smass('WZQQ') * smass(sName) * _extra_scaling
                _up = mtempl.get(shift=realshift)
                _down = mtempl.get(shift=-realshift)
                if badtemp_ma(_up[0]) or badtemp_ma(_down[0]):
                    print("Skipping sample {}, scale systematic would be empty".format(sName))
                    continue
                sample.setParamEffect(sys_scale, deepcopy(_up), deepcopy(_down), scale=1/_extra_scaling)
  
            if smear_syst and sName not in ['qcd']:
                _up = mtempl.get(smear=1 + SF[year]['smear_SF_ERR'] * _extra_scaling)
                _down = mtempl.get(smear=1 - SF[year]['smear_SF_ERR'] * _extra_scaling)
                if badtemp_ma(_up[0]) or badtemp_ma(_down[0]):
                    print("Skipping sample {}, scale systematic would be empty".format(sName))
                    continue
                sample.setParamEffect(sys_smear, _up, _down, scale=1/_extra_scaling)

            ch.addSample(sample)



       # sNameFake = 'fake'
       # fakes_sys_names = ['fakes_SF', 'fakes_DR']

       # fake_est_templ = get_templ(f, region, sNameFake)         
       # stype = rl.Sample.BACKGROUND

       # fakeSample = rl.TemplateSample(ch.name + '_' + sNameFake, stype, fake_est_templ)

      
       # if sNameFake == 'fake': 
       #     fakeSample.setParamEffect(fake_SF_uncert,1.5,None)

      
       #     for sys_nameFake in fakes_sys_names:   
       #         _up = get_templ(f, region, sNameFake, syst=sys_nameFake + "_Up")
       #         _dn = get_templ(f, region, sNameFake, syst=sys_nameFake + "_Down")
                
            # nominal = get_templ(f, region, sName, syst="fakes_nominal")
       #     fakeSample.setParamEffect(sys_shape_dict[sys_nameFake],_up[:-1], _dn[:-1])

       #     ch.addSample(fakeSample)




        data_obs = get_templ(f,region, 'data' )[:-1]    # Don't pass variances  
        ch.setObservation(data_obs)

        ch.autoMCStats( 
            channel_name=f"CMS_HWW_boosted_TopCR_{region}",
        )


 #for region in ['pass', 'fail']:
        #ch = rl.Channel("SR1{}{}".format(region, year))
        #model.addChannel(ch)
        #print('channel', ch)
         #failCh = model['ptbin{}fail{}'.format(ptbin, year)]




    failCh = model['TopCRfail{}'.format(year)]
    passCh = model['TopCRpass{}'.format(year)]

    tqqpass = passCh['TTbar']
    tqqfail = failCh['TTbar']
    
    tqqpass = passCh['TTbar']
    tqqfail = failCh['TTbar']
    stqqpass = passCh['SingleTop']
    stqqfail = failCh['SingleTop']

    sumPass = tqqpass.getExpectation(nominal=True).sum()
    sumFail = tqqfail.getExpectation(nominal=True).sum()
    sumPass += stqqpass.getExpectation(nominal=True).sum()
    sumFail += stqqfail.getExpectation(nominal=True).sum()
    tqqPF = sumPass / sumFail

    tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
    tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
    tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)
    stqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
    stqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    stqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
    stqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)


            

    with open('testModel_TopCR.root', 'wb') as fout:
    #with open("{}.pkl".format(model_name), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine('testModel_TopCR')

  

#******************************************************
ROOTFILE = '/home/jieun/VH-makeCards2/2016APV/2016APV_TopCR.root'
#******************************************************

import os
thisdir = os.getcwd()
print('thisdir', thisdir)
if "2016APV" in thisdir:
    year = "2016APV"
elif "2016" in thisdir:
    year = "2016"
elif "2017" in thisdir: 
    year = "2017"
elif "2018" in thisdir:
    year = "2018"

print('year', year)

dummy_rhalphabet(year)
