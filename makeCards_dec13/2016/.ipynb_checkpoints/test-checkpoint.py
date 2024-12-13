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
np.set_printoptions(precision=4)

def badtemp(hvalues, eps=0.0000001, mask=None):
    # Need minimum size & more than 1 non-zero bins
    tot = np.sum(hvalues[mask])
    count_nonzeros = np.sum(hvalues[mask] > 0)
    if (tot < eps) or (count_nonzeros < 2):
        return True
    else:
        return False

def get_bins():
    ptbins = np.array([40,60,80,100,120,140,160,180])
   
    return ptbins

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
        
    #if not muon:
     #   hist_name += "_SR1"
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
    #print('h_vals', h_vals)
    #print('var', h_variances)


    
    if np.any(h_vals <= .01):
        _invalid = h_vals <= .01
        h_vals[_invalid] = .1
        h_variances[_invalid] = 0.05
        
    if np.any(~np.isfinite(h_vals)):
        _invalid = ~np.isfinite(h_vals)
        h_vals[_invalid] = 0.001
        h_variances[_invalid] = 0.001
    h_key = 'msd'
    return (h_vals, h_edges, h_key, h_variances)


def dummy_rhalphabet( scale_syst=True, smear_syst=True,  systs=True,muonCR=False,year='2016',  opts=None):

    assert year in ['2016', '2017', '2018']

 
    sys_shape_dict = {}
    sys_shape_dict['weight_pileup'] = rl.NuisanceParameter('CMS_pileup_{}'.format(year), 'shape')
 
    #***********************************************************************************************************************
    msdbins = get_bins()
    #print('bins', msdbins)
    msd = rl.Observable('msd', msdbins)
    model = rl.Model("shapes" + year)  # build actual fit model now

    f = uproot.open(ROOTFILE)  #open file here
    #print('f.keys', f.keys())

    for region in ['pass', 'fail']:
        ch = rl.Channel("SR1{}{}".format(region, year))
        model.addChannel(ch)
        #print('channel', ch)

        include_samples = [
                   'WH', 
            'WQQ', 
       
                ]

      
      
      #just some functions/checking, ignore this section******************************
        from functools import partial
        # badtemp_ma = partial(badtemp, mask=mask)
        for sName in include_samples: # Remove empty samples
            print('sname', sName)
            templ = get_templ(f, region, sName)
            print('sName,region', sName, region, templ)
            if templ is None:
                print( 'Sample {} in region = {},not found in template file.' .format(sName, region))
                include_samples.remove(sName)
    #******************************
     
        for sName in include_samples:  #define signal and background samples
            templ = get_templ(f, region, sName)         
            if sName in ['WH', 'ZH', 'VBF', 'ggF', 'ttH']: 
                stype = rl.Sample.SIGNAL
            else:
                stype = rl.Sample.BACKGROUND

            sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)
            # Systematics
            #####################################################
            if not systs:  # Need at least one
                sample.setParamEffect(sys_lumi, lumi_dict[year])
            else:
                 
                sys_names = [ "weight_pileup",]
                
                for sys_name in sys_names:  
                    #now get the shapes
                    print('sys_name', sys_name)
                    _up = get_templ(f, region, sName, syst=sys_name + "_up")
                    _dn = get_templ(f, region, sName, syst=sys_name + "_down")
                    print('_up',_up[0],_up[3])
                    print('_down',_dn[0],_dn[3])
                    #print('_down',_dn.variances)
                      
                    if _up is None or _dn is None:
                        print('sample issue, no up or down for sample', sName)
                        continue

                    if (_up[3] ==_dn[3]).all():
                        sample.setParamEffect(sys_shape_dict[sys_name],_up[:-1])
                    else:
                    
                        sample.setParamEffect(sys_shape_dict[sys_name],_up[:-1], _dn[:-1])
                  #  sample.setParamEffect(sys_shape_dict[sys_name],_up[:-1])
                    #sample.setParamEffect(sys_shape_dict[sys_name],_up[:-1], _dn[:-1])
                    #sample.setParamEffect(sys_shape_dict[sys_name],(_up[:-1]/1), (_dn[:-1]/1))
    
         
            ch.addSample(sample)

    #for region in pass/fail (this loop is in second tab)
        #(add fakes here)

#def get_templ(f, region, sample, syst=None, muon=False, n
#if eff_scale_do < 0:
 #                                   eff_scale_do = eff_scale_up
    

#nominal[nominal == 0] = 1  # to avoid invalid value encountered in true_divide in "syst_up/nominal"
#            sample.setParamEffect(sys_value, (syst_up / nominal), (syst_do / nominal))

   


#***data***********************************************
        data_obs = get_templ(f,region, 'data' )[:-1]   
        ch.setObservation(data_obs)

        ch.autoMCStats( 
            channel_name=f"CMS_HWW_boosted_SR1_{region}",
        )


 #for region in ['pass', 'fail']:
        #ch = rl.Channel("SR1{}{}".format(region, year))
        #model.addChannel(ch)
        #print('channel', ch)
         #failCh = model['ptbin{}fail{}'.format(ptbin, year)]





    with open('testModel.root', 'wb') as fout:
    #with open("{}.pkl".format(model_name), "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine('testModel')

  

#******************************************************
ROOTFILE = '/home/jieun/VH-makeCards/2016/2016.root'
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
