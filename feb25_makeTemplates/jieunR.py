#!/usr/bin/python
import json
import pickle as pkl
import warnings
from typing import List

import numpy as np
import scipy
from hist import Hist
from systematics import sigs
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
from systematicsPass import get_systematic_dict, sigs
from utils import get_common_sample_name, get_finetuned_score, get_xsecweight
import math

warnings.filterwarnings("ignore", message="Found duplicate branch ")
#mass_binning =  [40,60,80,100,120,140,180]
#mass_binning =  [40,70,110,140,180]
mass_binning =  [40,180]
#mass_binning = [40,50,60,70,80,90,100,110,120,130,140,150,160,170,180]
#*************************************************************************************************************************************

WSF_2016 = [0.832,0.827,0.711]
WSF_2016APV = [0.850, 0.89,0.914]
WSF_2017 = [0.842,0.843,0.874]
WSF_2018 = [0.858,0.816,0.757]

ZSF_2016 = [0.989,0.990,1.001,0.999,0.992,1.058,1.048,1.056]
ZSF_2016APV = [1.004,1.013,1.016,1.010,0.984,1.044,1.002,1.086]
ZSF_2017 = [1.017,0.989,1.006,1.017,0.989,1.029,0.993,0.979]
ZSF_2018 = [0.967,0.969,0.989,0.972,0.964,1.026,1.010,1.002]

WSF_2016_Uncert_Up = [1.048,1.045,1.080]
WSF_2016_Uncert_Down = [0.954,0.959,0.929]
WSF_2016APV_Uncert_Up = [1.036,1.041,1.081]
WSF_2016APV_Uncert_Down = [0.955,0.959,0.929]
WSF_2017_Uncert_Up = [1.033,1.029,1.053]
WSF_2017_Uncert_Down = [0.968,0.971,0.949]
WSF_2018_Uncert_Up = [1.025,1.026,1.057]
WSF_2018_Uncert_Down = [0.974,0.975,0.959]
ZSF_2016_Uncert_Up = [1.052,1.037,1.028,1.031,1.036,1.057,1.071,1.120]
ZSF_2016_Uncert_Down = [0.949,0.964,0.969,0.973,0.963,0.955,0.955,0.905]
ZSF_2016APV_Uncert_Up = [1.026,1.040,1.034,1.046,1.060,1.065,1.077,1.138]
ZSF_2016APV_Uncert_Down = [0.972,0.959,0.97,0.961,0.938,0.941,0.929,0.852]
ZSF_2017_Uncert_Up = [1.049,1.035,1.035,1.032,1.034,1.028,1.025,1.037]
ZSF_2017_Uncert_Down = [0.952,0.965,0.968,0.963,0.966,0.98,0.976,0.959]
ZSF_2018_Uncert_Up = [1.038,1.027,1.025,1.028,1.031,1.027,1.022,1.041]
ZSF_2018_Uncert_Down = [0.958,0.966,0.968,0.965,0.961,0.977,0.978,0.959]

#ignore for now VH derived ones
VH_derivedSF_2017 = 0.955
VH_derivedUncertainty_2017_Up_Matched = 1.02
VH_derivedUncertainty_2017_Down_Matched = 0.98
VH_derivedmistagSF_2017 = 1.22
VH_derivedmistagSF_2017_up = 1.07
VH_derivedmistagSF_2017_down = 0.93

#try VH ones with mistag SF
def apply_V_SF_2017(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt > 250:
            return ZSF_2017[0]
        elif Vpt < 350 and Vpt > 300:
            return ZSF_2017[1]
        elif Vpt < 400 and Vpt > 350:
             return ZSF_2017[2]
        elif Vpt < 450 and Vpt > 400: 
            return ZSF_2017[3]
        elif Vpt < 500 and Vpt > 450:
            return ZSF_2017[4]
        elif Vpt < 600 and Vpt > 500:
            return ZSF_2017[5]
        elif Vpt < 800 and Vpt > 600:
            return ZSF_2017[6]
        elif Vpt < 2000 and Vpt > 800:
            return 1
    elif isMatched and isVqq and (nb_quarks != 2 or math.isnan(nb_quarks)):
        return VH_derivedSF_2017
    else: 
        return VH_derivedmistagSF_2017
        

def VH_applyV_Uncertainty_2017_up_Matched(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2017_Uncert_Up[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2017_Uncert_Up[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2017_Uncert_Up[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2017_Uncert_Up[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2017_Uncert_Up[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2017_Uncert_Up[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2017_Uncert_Up[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    elif isMatched and isVqq and (nb_quarks != 2 or math.isnan(nb_quarks)):
        return VH_derivedUncertainty_2017_Up_Matched
    else: 
        return 1

def VH_applyV_Uncertainty_2017_down_Matched(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2017_Uncert_Down[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2017_Uncert_Down[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2017_Uncert_Down[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2017_Uncert_Down[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2017_Uncert_Down[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2017_Uncert_Down[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2017_Uncert_Down[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    elif isMatched and isVqq and (nb_quarks != 2 or math.isnan(nb_quarks)):
        return VH_derivedUncertainty_2017_Down_Matched
    else: 
        return 1


def VH_applyV_Uncertainty_2017_up_Unmatched(isMatched,isVqq,nb_quarks,Vpt):
    if not isMatched:
        return VH_derivedmistagSF_2017_up
    else:
        return 1

def VH_applyV_Uncertainty_2017_down_Unmatched(isMatched,isVqq,nb_quarks,Vpt):
    if not isMatched:
        return VH_derivedmistagSF_2017_down
    else:
        return 1

#***JMAR SF **********************************************************************************
def applyJMAR_V_SF_2016APV(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2016APV[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2016APV[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2016APV[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2016APV[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2016APV[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2016APV[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2016APV[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: #isMatched and isVqq and (nb_quarks != 2 or math.isnan(nb_quarks)):
        if Vpt < 300 and Vpt >= 200:
            return WSF_2016APV[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2016APV[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2016APV[1]
        else:
            return 1



def applyJMAR_V_SF_2016(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2016[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2016[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2016[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2016[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2016[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2016[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2016[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: #isMatched and isVqq and (nb_quarks != 2 or math.isnan(nb_quarks)):
        if Vpt < 300 and Vpt >= 200:
            return WSF_2016[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2016[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2016[1]
        else:
            return 1
            
def applyJMAR_V_SF_2017(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2017[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2017[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2017[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2017[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2017[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2017[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2017[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2017[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2017[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2017[1]
        else:
            return 1

            
def applyJMAR_V_SF_2018(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2018[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2018[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2018[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2018[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2018[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2018[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2018[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2018[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2018[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2018[1]
        else:
            return 1


def applyV_Uncertainty_2016APV_up(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2016APV_Uncert_Up[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2016APV_Uncert_Up[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2016APV_Uncert_Up[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2016APV_Uncert_Up[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2016APV_Uncert_Up[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2016APV_Uncert_Up[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2016APV_Uncert_Up[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2016APV_Uncert_Up[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2016APV_Uncert_Up[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2016APV_Uncert_Up[2]
        else:
            return 1




def applyV_Uncertainty_2016_up(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2016_Uncert_Up[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2016_Uncert_Up[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2016_Uncert_Up[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2016_Uncert_Up[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2016_Uncert_Up[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2016_Uncert_Up[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2016_Uncert_Up[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2016_Uncert_Up[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2016_Uncert_Up[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2016_Uncert_Up[2]
        else:
            return 1
            
def applyV_Uncertainty_2017_up(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2017_Uncert_Up[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2017_Uncert_Up[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2017_Uncert_Up[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2017_Uncert_Up[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2017_Uncert_Up[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2017_Uncert_Up[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2017_Uncert_Up[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: # isMatched and isVqq and (nb_quarks != 2 or math.isnan(nb_quarks)):
        if Vpt < 300 and Vpt >= 200:
            return WSF_2017_Uncert_Up[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2017_Uncert_Up[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2017_Uncert_Up[2]
        else:
            return 1

def applyV_Uncertainty_2018_up(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2018_Uncert_Up[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2018_Uncert_Up[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2018_Uncert_Up[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2018_Uncert_Up[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2018_Uncert_Up[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2018_Uncert_Up[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2018_Uncert_Up[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: # isMatched and isVqq and (nb_quarks != 2 or math.isnan(nb_quarks)):
        if Vpt < 300 and Vpt >= 200:
            return WSF_2018_Uncert_Up[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2018_Uncert_Up[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2018_Uncert_Up[2]
        else:
            return 1


def applyV_Uncertainty_2016APV_down(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2016APV_Uncert_Down[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2016APV_Uncert_Down[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2016APV_Uncert_Down[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2016APV_Uncert_Down[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2016APV_Uncert_Down[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2016APV_Uncert_Down[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2016APV_Uncert_Down[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2016APV_Uncert_Down[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2016APV_Uncert_Down[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2016APV_Uncert_Down[2]
        else:
            return 1


    
def applyV_Uncertainty_2016_down(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2016_Uncert_Down[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2016_Uncert_Down[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2016_Uncert_Down[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2016_Uncert_Down[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2016_Uncert_Down[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2016_Uncert_Down[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2016_Uncert_Down[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2016_Uncert_Down[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2016_Uncert_Down[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2016_Uncert_Down[2]
        else:
            return 1


def applyV_Uncertainty_2017_down(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2017_Uncert_Down[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2017_Uncert_Down[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2017_Uncert_Down[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2017_Uncert_Down[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2017_Uncert_Down[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2017_Uncert_Down[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2017_Uncert_Down[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2017_Uncert_Down[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2017_Uncert_Down[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2017_Uncert_Down[2]
        else:
            return 1

def applyV_Uncertainty_2018_down(isMatched,isVqq,nb_quarks,Vpt):
    if isMatched and isVqq and nb_quarks == 2:
        if Vpt < 300 and Vpt >= 250:
            return ZSF_2018_Uncert_Down[0]
        elif Vpt < 350 and Vpt >= 300:
            return ZSF_2018_Uncert_Down[1]
        elif Vpt < 400 and Vpt >= 350:
             return ZSF_2018_Uncert_Down[2]
        elif Vpt < 450 and Vpt >= 400: 
            return ZSF_2018_Uncert_Down[3]
        elif Vpt < 500 and Vpt >= 450:
            return ZSF_2018_Uncert_Down[4]
        elif Vpt < 600 and Vpt >= 500:
            return ZSF_2018_Uncert_Down[5]
        elif Vpt < 800 and Vpt >= 600:
            return ZSF_2018_Uncert_Down[6]
        elif Vpt < 2000 and Vpt >= 800:
            return 1
    else: 
        if Vpt < 300 and Vpt >= 200:
            return WSF_2018_Uncert_Down[0]
        elif Vpt < 400 and Vpt >= 300:
            return WSF_2018_Uncert_Down[1]
        elif Vpt < 2000 and Vpt >= 400:
            return WSF_2018_Uncert_Down[2]
        else:
            return 1


#*********************************************************

#2018 
nom_trig = [  0.97316853, 1.00480192, 0.99773121, 1.00115747, 0.97460874,1.02083724, 1.00569659, 0.99931007, 0.97253711, 1.02816501, 0.9582612 , 0.99847442, 0.9663244 , 1.00026573, 0.97930598]
up_trig = [1.04406395, 1.0161899 , 1.00818656, 1.01453322, 1.0504374,1.05290024, 1.01810422, 1.0107254 , 0.98913642, 1.06335367,0.99717576, 1.0172449 , 0.98301419, 1.02056413, 1.01756531]
down_trig = [0.94615435, 1.00361001, 0.99175464, 0.99613031, 0.9455259, 1.00181298, 0.99597402, 0.98982106, 0.95861722, 1.00558644, 0.92292573, 0.98087248, 0.9504349 , 0.9812107 , 0.94474428]

#2017
nom_trig2017 = [1.01467505, 0.99558953, 0.99596328, 0.97266757, 1.01680672,0.91771569, 0.97330338, 0.9628086 , 0.94945886, 0.93074193,0.88346762, 0.93986709, 0.95049357, 0.96395947, 0.96216591 ]
up_trig2017 = [1.08202419, 1.01316206, 1.00933387, 0.99761035, 1.09885624, 0.98177762, 0.99283257, 0.97860141, 0.96964921, 0.99310961, 0.93970839, 0.96592046, 0.97095216, 0.98912634, 1.01346021]
down_trig2017 = [1.00919919, 0.9891299 , 0.98928763, 0.95716839, 1.01091358, 0.86905073, 0.95735603, 0.94920227, 0.93221049, 0.88617271,0.83084485, 0.91516138, 0.93099722, 0.94018822, 0.91551657 ]

#2016
nom_trig2016 = [1.00984252, 0.98759196, 0.99880955, 0.9905723 , 1.00215983,0.98220745, 1.01275276, 0.98584499, 1.00147192, 1.02677684,0.91866763, 1.00740741, 1.02089672, 1.04459947, 0.96555385 ]
up_trig2016 = [1.08470782, 1.01762475, 1.01440987, 1.01465713, 1.09484157,1.03099084, 1.02538254, 1.00007443, 1.01757504, 1.06379468,0.98139906, 1.03550959, 1.04627989, 1.07175043, 1.02924617 ]
down_trig2016 = [1.00555572, 0.97653751, 0.9930635 , 0.98174029, 1.00036919, 0.95711356, 1.00545487, 0.97463365, 0.99051333, 1.0122556,0.85933502, 0.98129079, 0.99705125, 1.01968035, 0.90658386 ]


def applyTriggerSF_2018(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig[14]
    else:
        return 1


def applyTriggerSF_2017(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig2017[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig2017[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig2017[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig2017[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig2017[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig2017[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig2017[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig2017[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig2017[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig2017[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig2017[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig2017[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig2017[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig2017[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig2017[14]
    else:
        return 1

def applyTriggerSF_2016(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig2016[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig2016[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig2016[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig2016[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig2016[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig2016[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig2016[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig2016[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig2016[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig2016[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return nom_trig2016[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return nom_trig2016[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return nom_trig2016[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return nom_trig2016[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return nom_trig2016[14]
    else:
        return 1

def applyTriggerSF_up_2018(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig[14]
    else:
        return 1

def applyTriggerSF_up_2017(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig2017[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig2017[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig2017[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig2017[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig2017[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig2017[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig2017[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig2017[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig2017[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig2017[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig2017[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig2017[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig2017[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig2017[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig2017[14]
    else:
        return 1

def applyTriggerSF_up_2016(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig2016[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig2016[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig2016[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig2016[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig2016[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig2016[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig2016[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig2016[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig2016[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig2016[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return up_trig2016[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return up_trig2016[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return up_trig2016[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return up_trig2016[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return up_trig2016[14]
    else:
        return 1



def applyTriggerSF_down_2018(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig[14]
    else:
        return 1

def applyTriggerSF_down_2017(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig2017[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig2017[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig2017[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig2017[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig2017[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig2017[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig2017[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig2017[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig2017[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig2017[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig2017[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig2017[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig2017[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig2017[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig2017[14]
    else:
        return 1

def applyTriggerSF_down_2016(lep_pt,lep_eta):
    if lep_pt < 2000 and lep_pt > 200 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig2016[0]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig2016[1]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig2016[2]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig2016[3]
    elif lep_pt < 2000 and lep_pt > 200 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig2016[4]
    if lep_pt < 200 and lep_pt > 120 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig2016[5]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig2016[6]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig2016[7]
    elif lep_pt < 200 and lep_pt > 120 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig2016[8]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig2016[9]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -1.5 and lep_eta > -2.5:
        return down_trig2016[10]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < -0.5 and lep_eta > -1.5:
        return down_trig2016[11]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 0.5 and lep_eta > -0.5:
        return down_trig2016[12]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 1.5 and lep_eta > 0.5:
        return down_trig2016[13]
    elif lep_pt < 120 and lep_pt > 30 and lep_eta < 2.5 and lep_eta > 1.5:
        return down_trig2016[14]
    else:
        return 1



#*************************************************************************************************************************************
def fix_neg_yields(h):
    """
    Will set the bin yields of a process to 0 if the nominal yield is negative, and will
    set the yield to 0 for the full Systematic axis.
    """
    for region in h.axes["Region"]:
        for sample in h.axes["Sample"]:
            neg_bins = np.where(h[{"Sample": sample, "Systematic": "pass_nominal", "Region": region}].values() < 0)[0]

            if len(neg_bins) > 0:
                print('got neg bins')
                print(f"{region}, {sample}, has {len(neg_bins)} bins with negative yield.. will set them to 0")

                sample_index = np.argmax(np.array(h.axes["Sample"]) == sample)
                region_index = np.argmax(np.array(h.axes["Region"]) == region)

                for neg_bin in neg_bins:
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].value = 1e-3
                    h.view(flow=True)[sample_index, :, region_index, neg_bin + 1].variance = 1e-3

#*****************************************************************************************************************
def get_templates(years, channels, samples, samples_dir, regions_sel, model_path, add_fake=False):

    # add extra selections to preselection
    presel = { "mu": { "fj_mass": "fj_mass < 180",}, "ele": {"fj_mass": "fj_mass <180",}, }
   
    hists = hist2.Hist( hist2.axis.StrCategory([], name="Sample", growth=True), hist2.axis.StrCategory([], name="Systematic", growth=True), 
                       hist2.axis.StrCategory([], name="Region", growth=True),
                        hist2.axis.Variable(
            #list(range(55, 255, mass_binning)),
            mass_binning, name="mass_observable", label=r"V reconstructed mass [GeV]", overflow=True,
        ), storage=hist2.storage.Weight(),
    )

    SYST_DICT = get_systematic_dict(years)

    for year in years:  # e.g. 2018, 2017, 2016APV, 2016
        for ch in channels:  # e.g. mu, ele
            logging.info(f"Processing year {year} and {ch} channel")
            with open("../../fileset/luminosity.json") as f:
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

                data["THWW"] = get_finetuned_score(data, model_path)
                data = data[data.columns.drop(list(data.filter(regex="hidNeuron")))]

                for selection in presel[ch]:
                    logging.info(f"Applying {selection} selection on {len(data)} events")
                    data = data.query(presel[ch][selection])

#********************************************************************************************************                  
                    if not is_data:
                        data['met_pt_UES_up'] = data['ues_up']
                        data['met_pt_UES_down'] = data['ues_down'] #to do: reame in processor to elim. this step

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
                        data['temp_JES_Total_down'] = data['met_pt_JES_Total_down']
                        data['temp_JES_Total_up'] = data['met_pt_JES_Total_up']
                        data[['met_pt_JES_Total_down', 'met_pt_JES_Total_up']]= data[['temp_JES_Total_up', 'temp_JES_Total_down']]

                        if year == '2018':
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

                        elif year == '2017':
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

                        elif year == '2016' or year == '2016APV':
                            data['temp_JES_BBEC1_2016_down'] = data['met_pt_JES_BBEC1_2016_down']
                            data['temp_JES_BBEC1_2016_up'] = data['met_pt_JES_BBEC1_2016_up']
                            data[['met_pt_JES_BBEC1_2016_down', 'met_pt_JES_BBEC1_2016_up']]= data[['temp_JES_BBEC1_2016_up', 'temp_JES_BBEC1_2016_down']]
                            data['temp_JES_RelativeSample_2016_down'] = data['met_pt_JES_RelativeSample_2016_down']
                            data['temp_JES_RelativeSample_2016_up'] = data['met_pt_JES_RelativeSample_2016_up']
                            data[['met_pt_JES_RelativeSample_2016_down', 'met_pt_JES_RelativeSample_2016_up']]= data[['temp_JES_RelativeSample_2016_up', 'temp_JES_RelativeSample_2016_down']]
                            data['temp_JES_EC2_2016_down'] = data['met_pt_JES_EC2_2016_down']
                            data['temp_JES_EC2_2016_up'] = data['met_pt_JES_EC2_2016_up']
                            data[['met_pt_JES_EC2_2016_down', 'met_pt_JES_EC2_2016_up']]= data[['temp_JES_EC2_2016_up', 'temp_JES_EC2_2016_down']]
                            data['temp_JES_HF_2016_down'] = data['met_pt_JES_HF_2016_down']
                            data['temp_JES_HF_2016_up'] = data['met_pt_JES_HF_2016_up']
                            data[['met_pt_JES_HF_2016_down', 'met_pt_JES_HF_2016_up']]= data[['temp_JES_HF_2016_up', 'temp_JES_HF_2016_down']]
                            data['temp_JES_Absolute_2016_down'] = data['met_pt_JES_Absolute_2016_down']
                            data['temp_JES_Absolute_2016_up'] = data['met_pt_JES_Absolute_2016_up']
                            data[['met_pt_JES_Absolute_2016_down', 'met_pt_JES_Absolute_2016_up']]= data[['temp_JES_Absolute_2016_up', 'temp_JES_Absolute_2016_down']]

#********************************************************************************************************
                # get the xsecweight
                xsecweight, sumgenweights, sumpdfweights, sumscaleweights = get_xsecweight( pkl_files, year, sample, sample_to_use, is_data, luminosity)

                for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
           
#**************************************************************************************
                    df = data.copy()
                    logging.info(f"Applying {region} selection on {len(df)} events")
                    df = df.query(region_sel)
                    logging.info(f"Will fill the histograms with the remaining {len(df)} events")

                    # ------------------- Nominal -------------------
                    if is_data:
                        nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                    else:
                        nominal = df[f"weight_{ch}"] * xsecweight * df["weight_btag"] 
                        print('above nominal', nominal)
#*******************************************
#ParT SF                        
                    if sample_to_use in ["ggF", "VBF", "WH", "ZH", "ttH"]:
                        nominal *= 0.984
                    else:
                        nominal *= 1
#*******************************************
#JMAR SF BY YEAR
                    if sample_to_use in ["ZH", "WH", "ggF", "VBF", "ttH", "TTbar", "Diboson", "SingleTop"]:
                        if year == '2017':
                            data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2017(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                            nominal *= data['tempZ']
                        if year == '2016':
                            data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                            nominal *= data['tempZ']
                        if year == '2016APV':
                            data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016APV(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                            nominal *= data['tempZ']
                        if year == '2018':
                            data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2018(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                            nominal *= data['tempZ']
                            
#*********************************************************************************************************************************
                        #for trigger SF 
                        if ch == 'ele':
                            if year == '2018':
                                df['temp'] = df.apply(lambda row:applyTriggerSF_2018(row['lep_pt'],row['lep_eta']),axis=1)
                                nominal *=  df['temp']
                                df.drop(columns=['temp'],axis=1)
                            elif year == '2017':
                                df['temp'] = df.apply(lambda row:applyTriggerSF_2017(row['lep_pt'],row['lep_eta']),axis=1)
                                nominal *=  df['temp']
                                df.drop(columns=['temp'],axis=1)
                            else:
                                df['temp'] = df.apply(lambda row:applyTriggerSF_2016(row['lep_pt'],row['lep_eta']),axis=1)
                                nominal *=  df['temp']
                                df.drop(columns=['temp'],axis=1)        
                #************
                        if sample_to_use == "TTbar":
                            nominal *= df["top_reweighting"]
                    ###################################
                        if sample_to_use == "EWKvjets":
                            threshold = 20
                            df = df[nominal < threshold]
                            nominal = nominal[nominal < threshold]
                    ###################################

                    hists.fill( Sample=sample_to_use, Systematic="pass_nominal", Region=region, mass_observable=df["fj_mass"], weight=nominal,)

                    #ABOVE IS ALL WEIGHTS THAT NOMINAL SHOULD HAVE and that is compared later on; need above if I change def. of nominal



#*********************************************************************************************************************************
                    
                    for syst, (yrs, smpls, var) in SYST_DICT["psrad"].items():
                        print('syst', syst)
                        if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                            shape_up = df[var[ch] + "Up"]  * xsecweight * df["weight_btag"] 
                            shape_down = df[var[ch] + "Down"] * xsecweight * df["weight_btag"]      

                            if sample_to_use == "TTbar":
                                shape_up *= df["top_reweighting"]
                                shape_down *= df["top_reweighting"]
                     

#try fixing the shape up and shape down to have the nominal weight
                            if sample_to_use in ["ggF", "VBF", "WH", "ZH", "ttH"]:
                                shape_up *= 0.984
                                shape_down *= 0.984
                 
            #*******************************************
            #JMAR SF BY YEAR
                            if sample_to_use in ["ZH", "WH", "ggF", "VBF", "ttH", "TTbar", "Diboson", "SingleTop"]:
                                if year == '2017':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2017(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                                if year == '2016':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                                if year == '2016APV':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016APV(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                                if year == '2018':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2018(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                        
    #*********************************************************************************************************************************
                            #for trigger SF 
                            if ch == 'ele':
                                if year == '2018':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2018(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up *=  df['temp']
                                    shape_down *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                elif year == '2017':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2017(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up *=  df['temp']
                                    shape_down *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                else:
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2016(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up *=  df['temp']
                                    shape_down *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)        
    
                        
                        ###################################

                        
                
                        
                        else:
                            shape_up = nominal
                            shape_down = nominal
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_{sample_to_use}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up,)
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_{sample_to_use}_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down, )


#*********************************************************************************************************************************
#add these lines for the new EWK correction for signal only 
                    #for syst, (yrs, smpls, var) in SYST_DICT["EWK"].items():  #this works
                     #   if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                    up, down = nominal.copy(), nominal.copy()
                    if sample_to_use in ["ZH", "WH", "VBF", "ttH"]:
                        
                        print('fj genHT', data["fj_genH_pt"])   #this is 9308 but the others are 575
                        print('nominal', nominal)
                        msk = df["fj_genH_pt"] > 400
                       

                        shape_up = np.where(msk, nominal / df["EW_weight"], nominal)
                        shape_down = np.where(msk, nominal * df["EW_weight"], nominal)
                        
                        hists.fill( Sample=sample_to_use, Systematic="pass_EW_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up,)
                        hists.fill( Sample=sample_to_use, Systematic="pass_EW_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down, )
           

#*********************************************************************************************************************************


                    
                    # ------------------- Common systematics  -------------------

#nominal has event weight, but event weight already has pileup weight in it since it is a product of various event weights, so i guess that is why farouk only has xsec weight.
                    for syst, (yrs, smpls, var) in SYST_DICT["common"].items():
                        #print('syst', syst)
                        if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                            shape_up = df[var[ch] + "Up"] * xsecweight * df["weight_btag"]  #*  df[f"weight_{ch}"]
                            shape_down = df[var[ch] + "Down"] * xsecweight * df["weight_btag"] #*  df[f"weight_{ch}"]
                            
                            if sample_to_use == "TTbar":
                                shape_up *= df["top_reweighting"]
                                shape_down *= df["top_reweighting"]
                     

#try fixing the shape up and shape down to have the nominal weight
                            if sample_to_use in ["ggF", "VBF", "WH", "ZH", "ttH"]:
                                shape_up *= 0.984
                                shape_down *= 0.984
                 
            #*******************************************
            #JMAR SF BY YEAR
                            if sample_to_use in ["ZH", "WH", "ggF", "VBF", "ttH", "TTbar", "Diboson", "SingleTop",]:
                                if year == '2017':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2017(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                                if year == '2016':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                                if year == '2016APV':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016APV(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                                if year == '2018':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2018(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up *= data['tempZ']
                                    shape_down *= data['tempZ']
                        
    #*********************************************************************************************************************************
                            #for trigger SF 
                            if ch == 'ele':
                                if year == '2018':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2018(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up *=  df['temp']
                                    shape_down *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                elif year == '2017':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2017(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up *=  df['temp']
                                    shape_down *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                else:
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2016(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up *=  df['temp']
                                    shape_down *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)        
    
                        
                        ###################################

                        else:
                            shape_up = nominal
                            shape_down = nominal
                     

                        
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up, )
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down, )

       #*****************************************************************************************************************
                    for syst, (yrs, smpls, var) in SYST_DICT["btag1"].items():  #this works
                        if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                            shape_up = df[var[ch] + "Up"]  * nominal # xsecweight * df[f"weight_{ch}"] 
                            shape_down = df[var[ch] + "Down"]  * nominal #xsecweight * df[f"weight_{ch}"] 
                        else:                       
                            shape_up = df['fj_mass']
                            shape_down = df['fj_mass']
          
                        hists.fill(Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up,)
                        hists.fill(Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down,)

                   #****************************************btag by year:
                    for syst, (yrs, smpls, var) in SYST_DICT["btag2"].items(): 
                        if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                            shape_up = df[var[ch] + "Up"] * nominal #* xsecweight * df[f"weight_{ch}"]  
                            shape_down = df[var[ch] + "Down"] * nominal # xsecweight * df[f"weight_{ch}"]          
                        else:                       
                            shape_up = df['fj_mass']
                            shape_down = df['fj_mass']
                  #      if sample_to_use == "TTbar":
                   #             shape_up *= df["top_reweighting"]
                    #            shape_down *= df["top_reweighting"]
                        hists.fill(Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up)
                        hists.fill(Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down)

  #*****************************************************************************************************************
                    

                    

#for up/down uncertainty JMAR WTAG SF ****************************************************************
                    for syst, (yrs, smpls, var) in SYST_DICT["Vtag"].items(): 
                            if year == '2018':
                                if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                    df['tempV_up'] = df.apply(lambda row:applyV_Uncertainty_2018_up(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    df["tempV_down"] = df.apply(lambda row:applyV_Uncertainty_2018_down(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up=nominal*df['tempV_up']
                                    shape_down=nominal*df['tempV_down']
                                    df.drop(columns=['tempV_up'],axis=1)
                                    df.drop(columns=['tempV_down'],axis=1)
                                else:
                                        shape_up = nominal
                                        shape_down = nominal
                                hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up )
                                hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"],weight=shape_down )
            
                            if year == '2017':
                                if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                    df['tempV_up'] = df.apply(lambda row:applyV_Uncertainty_2017_up(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    df["tempV_down"] = df.apply(lambda row:applyV_Uncertainty_2017_down(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    shape_up=nominal*df['tempV_up']
                                    shape_down=nominal*df['tempV_down']
                                    df.drop(columns=['tempV_up'],axis=1)
                                    df.drop(columns=['tempV_down'],axis=1)
                                else:
                                        shape_up = nominal
                                        shape_down = nominal
                                hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up )
                                hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"],weight=shape_down )

                            if year == '2016':
                                    if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                        df['tempV_up'] = df.apply(lambda row:applyV_Uncertainty_2016_up(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                        df["tempV_down"] = df.apply(lambda row:applyV_Uncertainty_2016_down(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                        shape_up=nominal*df['tempV_up']
                                        shape_down=nominal*df['tempV_down']
                                        df.drop(columns=['tempV_up'],axis=1)
                                        df.drop(columns=['tempV_down'],axis=1)
                                    else:
                                            shape_up = nominal
                                            shape_down = nominal
                                    hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up )
                                    hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"],weight=shape_down )

                            if year == '2016APV':
                                    if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                        df['tempV_up'] = df.apply(lambda row:applyV_Uncertainty_2016APV_up(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                        df["tempV_down"] = df.apply(lambda row:applyV_Uncertainty_2016APV_down(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                        shape_up=nominal*df['tempV_up']
                                        shape_down=nominal*df['tempV_down']
                                        df.drop(columns=['tempV_up'],axis=1)
                                        df.drop(columns=['tempV_down'],axis=1)
                                    else:
                                            shape_up = nominal
                                            shape_down = nominal
                                    hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up )
                                    hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"],weight=shape_down )
                    

            

                    # ------------------- PDF acceptance -------------------
                    """
                    For the PDF acceptance uncertainty:
                    - store 103 variations. 0-100 PDF values, The last two values: alpha_s variations.
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

                    hists.fill(Sample=sample_to_use, Systematic=f"pass_weight_pdf_acceptance_{sample_to_use}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up, )
                    hists.fill( Sample=sample_to_use, Systematic=f"pass_weight_pdf_acceptance_{sample_to_use}_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down,)

                    
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
                    if (sample_to_use in sigs + ["WJetsLNu", "TTbar", "SingleTop"]) and (sample != "ST_s-channel_4f_hadronicDecays" ):
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
                        scaleweights = np.swapaxes(np.array(scaleweights), 0, 1  )  # so that the shape is (# events, variation)
                        # TODO: debug
                        shape_up = nominal * np.max(scaleweights, axis=1) / scaleweight_4
                        shape_down = nominal * np.min(scaleweights, axis=1) / scaleweight_4
                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill(Sample=sample_to_use, Systematic=f"pass_weight_qcd_scale_{sample_to_use}_up", Region=region, mass_observable=df["fj_mass"],  weight=shape_up,)
                    hists.fill( Sample=sample_to_use, Systematic=f"pass_weight_qcd_scale_{sample_to_use}_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down, )



            
#*****************************histo for up and down trigger scale factor for electron only
                    for syst, (yrs, smpls, var) in SYST_DICT["TRIGGER_systs"].items(): #this dictionary only applies it to electron
                        if year == '2018':
                            if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                    df['weight_up'] = df.apply(lambda row:applyTriggerSF_up_2018(row['lep_pt'],row['lep_eta']),axis=1)
                                    df["weight_down"] = df.apply(lambda row:applyTriggerSF_down_2018(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up=nominal*df['weight_up']
                                    shape_down=nominal*df['weight_down']
                                    df.drop(columns=['weight_up'],axis=1)
                                    df.drop(columns=['weight_down'],axis=1)
                       
                            else:
                                    shape_up = nominal
                                    shape_down = nominal

                        elif year == '2017':
                            if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                    df['weight_up'] = df.apply(lambda row:applyTriggerSF_up_2017(row['lep_pt'],row['lep_eta']),axis=1)
                                    df["weight_down"] = df.apply(lambda row:applyTriggerSF_down_2017(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up=nominal*df['weight_up']
                                    shape_down=nominal*df['weight_down']
                                    df.drop(columns=['weight_up'],axis=1)
                                    df.drop(columns=['weight_down'],axis=1)                                 
                            else:
                                    shape_up = nominal
                                    shape_down = nominal

                        else:
                            if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                    df['weight_up'] = df.apply(lambda row:applyTriggerSF_up_2016(row['lep_pt'],row['lep_eta']),axis=1)
                                    df["weight_down"] = df.apply(lambda row:applyTriggerSF_down_2016(row['lep_pt'],row['lep_eta']),axis=1)
                                    shape_up=nominal*df['weight_up']
                                    shape_down=nominal*df['weight_down']
                                    df.drop(columns=['weight_up'],axis=1)
                                    df.drop(columns=['weight_down'],axis=1)                           
                            else:
                                    shape_up = nominal
                                    shape_down = nominal
                        
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up )
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=df["fj_mass"],weight=shape_down )
    
#*****************************************************************************************
                      
                    # ------------------- Top pt reweighting systematic  -------------------
                    if sample_to_use == "TTbar":
                        # first remove the reweighting effect
                        nominal_noreweighting = nominal / df["top_reweighting"]

                        shape_up = nominal_noreweighting * (df["top_reweighting"] ** 2)  # "up" is twice the correction
                        shape_down = nominal_noreweighting  # "down" is no correction
                    else:
                        shape_up = nominal
                        shape_down = nominal

                    hists.fill( Sample=sample_to_use, Systematic="pass_top_reweighting_up", Region=region, mass_observable=df["fj_mass"], weight=shape_up,)
                    hists.fill( Sample=sample_to_use, Systematic="pass_top_reweighting_down", Region=region, mass_observable=df["fj_mass"], weight=shape_down,)


                
#JMR/JMS**********************************************************************************

#["ZH", "WH", "ggF", "VBF", "ttH", "TTbar", "Diboson", "SingleTop"]:
                    
                    for syst, (yrs, smpls, var) in SYST_DICT["JEC_systs_mass1"].items():
                        #if (sample_to_use in smpls2) and (year in yrs) and (ch in var):  
                        if (sample_to_use in ["ZH", "WH", "ggF", "VBF", "ttH"]) and (year in yrs) and (ch in var):  
                            shape_up = df["fj_mass" + var[ch] + "_up"]  #* nominal
                            shape_down = df["fj_mass" + var[ch] + "_down"] #*nominal                     
                        else:
                            shape_up = df["fj_mass"]
                            shape_down = df["fj_mass"]
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=shape_up, weight=nominal )
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=shape_down, weight=nominal )
          
                    for syst, (yrs, smpls, var) in SYST_DICT["JEC_systs_mass2"].items():
                        #if (sample_to_use in smpls2) and (year in yrs) and (ch in var):  
                        if (sample_to_use in ["TTbar", "Diboson", "SingleTop"]) and (year in yrs) and (ch in var):  
                            shape_up = df["fj_mass" + var[ch] + "_up"]  #* nominal
                            shape_down = df["fj_mass" + var[ch] + "_down"] #*nominal 
                            #print('sahpe up', shape_up)                
                        else:
                            shape_up = df["fj_mass"]
                            shape_down = df["fj_mass"]
                    
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=shape_up, weight=nominal )
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=shape_down, weight=nominal )

                    for syst, (yrs, smpls, var) in SYST_DICT["JEC_systs_mass3"].items():
                        #if (sample_to_use in smpls2) and (year in yrs) and (ch in var):  
                        if (sample_to_use in ["DYJets", "WJetsLNu"]) and (year in yrs) and (ch in var):  
                            shape_up = df["fj_mass" + var[ch] + "_up"]  #* nominal
                            shape_down = df["fj_mass" + var[ch] + "_down"] #*nominal 
                            #print('sahpe up', shape_up)
                        else:
                            shape_up = df["fj_mass"]
                            shape_down = df["fj_mass"]
                    
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_up", Region=region, mass_observable=shape_up, weight=nominal )
                        hists.fill( Sample=sample_to_use, Systematic=f"{syst}_down", Region=region, mass_observable=shape_down, weight=nominal )
          
            

#*************************Need to put back the previous defs of nominal!!!!!!!    
                   # ------------------- individual sources of JES -------------------
                #HERE IS A NEW REGION SELECTION, and NEW definition of nominal **********************************NEW DEF OF NOMINAL BE CAREFUL
                """We apply the jet pt cut on the up/down variations. Must loop over systematics first."""
                for syst, (yrs, smpls, var) in SYST_DICT["JEC"].items():
                    for variation in ["up", "down"]:
                        for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                            if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                region_sel = region_sel.replace("fj_pt", "fj_pt" + var[ch] + f"_{variation}")
                                region_sel = region_sel.replace("met_pt", "met_pt_" + var[ch] + f"_{variation}")
                                region_sel = region_sel.replace("numberBJets_Medium_OutsideFatJets", "numberBJets_" + var[ch] + f"_{variation}")

                            df = data.copy()
                            df = df.query(region_sel)
                            # ------------------- Nominal -------------------
                            if is_data:
                                nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                            else:
                                nominal = df[f"weight_{ch}"] * xsecweight * df["weight_btag"] 
                      
                            if sample_to_use in ["ggF", "VBF", "WH", "ZH", "ttH"]:
                                nominal *= 0.984
                            else:
                                nominal *= 1
            #*******************************************
            #JMAR SF BY YEAR
                            if sample_to_use in ["ZH", "WH", "ggF", "VBF", "ttH", "TTbar", "Diboson", "SingleTop"]:
                                if year == '2017':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2017(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                                if year == '2016':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                                if year == '2016APV':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016APV(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                                if year == '2018':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2018(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                        
    #*********************************************************************************************************************************
                            #for trigger SF 
                            if ch == 'ele':
                                if year == '2018':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2018(row['lep_pt'],row['lep_eta']),axis=1)
                                    nominal *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                elif year == '2017':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2017(row['lep_pt'],row['lep_eta']),axis=1)
                                    nominal *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                else:
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2016(row['lep_pt'],row['lep_eta']),axis=1)
                                    nominal *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)        
                    #************
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

                            hists.fill( Sample=sample_to_use, Systematic=f"{syst}_{variation}", Region=region, mass_observable=shape_variation, weight=nominal,)


#AGAIN, need to put all the nominal weights again because redefining nominal
                
                for syst, (yrs, smpls, var) in SYST_DICT["UES_systs"].items():
                    #print('variation', variation)
                    for variation in ["up", "down"]:
                        for region, region_sel in regions_sel.items():  # e.g. pass, fail, top control region, etc.
                            if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                                region_sel = region_sel.replace("met_pt", "met_pt_" + var[ch] + f"_{variation}")
                               # print('region_sel', region_sel)
                            df = data.copy()
                            df = df.query(region_sel)
                            # ------------------- Nominal -------------------
                            if is_data:
                                nominal = np.ones_like(df["fj_pt"])  # for data (nominal is 1)
                            else:
                                nominal = df[f"weight_{ch}"] * xsecweight * df["weight_btag"]

            #*******************************************
                            if sample_to_use in ["ggF", "VBF", "WH", "ZH", "ttH"]:
                                nominal *= 0.984
                            else:
                                nominal *= 1
            #*******************************************
            #JMAR SF BY YEAR
                            if sample_to_use in ["ZH", "WH", "ggF", "VBF", "ttH", "TTbar", "Diboson", "SingleTop"]:
                                if year == '2017':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2017(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                                if year == '2016':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                                if year == '2016APV':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2016APV(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                                if year == '2018':
                                    data['tempZ'] = data.apply(lambda row:applyJMAR_V_SF_2018(row['fj_V_isMatched'],row['fj_isV_2q'],row['fj_nbquarks'], row['fj_pt']),axis=1)
                                    nominal *= data['tempZ']
                        
    #*********************************************************************************************************************************
                            #for trigger SF 
                            if ch == 'ele':
                                if year == '2018':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2018(row['lep_pt'],row['lep_eta']),axis=1)
                                    nominal *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                elif year == '2017':
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2017(row['lep_pt'],row['lep_eta']),axis=1)
                                    nominal *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)
                                else:
                                    df['temp'] = df.apply(lambda row:applyTriggerSF_2016(row['lep_pt'],row['lep_eta']),axis=1)
                                    nominal *=  df['temp']
                                    df.drop(columns=['temp'],axis=1)        
                    #************
                            if sample_to_use == "TTbar":
                                nominal *= df["top_reweighting"]
                        ###################################
                            if sample_to_use == "EWKvjets":
                                threshold = 20
                                df = df[nominal < threshold]
                                nominal = nominal[nominal < threshold]
                        ###################################


                        #    if (sample_to_use in smpls) and (year in yrs) and (ch in var):
                        #        shape_variation = df["fj_mass"]
                                
                        #    else:
                        #        shape_variation = df["fj_mass"]

                            hists.fill( Sample=sample_to_use, Systematic=f"{syst}_{variation}", Region=region, mass_observable=df['fj_mass'], weight=nominal,)

    
    return hists



def getPassList(sample):
    systematicsList_pass = ['pass_nominal', 'pass_weight_pileup_up' , 'pass_weight_pileup_down', 'pass_weight_pileup_id_up' , 'pass_weight_pileup_id_down',
                        f'pass_weight_pdf_acceptance_{sample}_up', f'pass_weight_pdf_acceptance_{sample}_down',
f'pass_weight_qcd_scale_{sample}_up', f'pass_weight_qcd_scale_{sample}_down',
'pass_top_reweighting_up', 'pass_top_reweighting_down',
'pass_weight_d1K_NLO_up', 'pass_weight_d1K_NLO_down', 'pass_weight_d2K_NLO_up', 'pass_weight_d2K_NLO_down', 'pass_weight_d3K_NLO_up', 
'pass_weight_d3K_NLO_down', 'pass_weight_d1kappa_EW_up', 'pass_weight_d1kappa_EW_down', 'pass_weight_W_d2kappa_EW_up', 'pass_weight_W_d2kappa_EW_down', 
'pass_weight_W_d3kappa_EW_up', 'pass_weight_W_d3kappa_EW_down', 'pass_weight_Z_d2kappa_EW_up', 'pass_weight_Z_d2kappa_EW_down', 
'pass_weight_Z_d3kappa_EW_up', 'pass_weight_Z_d3kappa_EW_down',
'pass_weight_btagSFlightCorrelated_up', 'pass_weight_btagSFlightCorrelated_down', 'pass_weight_btagSFbcCorrelated_up','pass_weight_btagSFbcCorrelated_down', 
'pass_weight_btagSFlight_up', 'pass_weight_btagSFlight_down', 'pass_weight_btagSFbc_up', 'pass_weight_btagSFbc_down',
'pass_weight_ele_id_up', 'pass_weight_ele_id_down', 'pass_weight_ele_reco_up', 'pass_weight_ele_reco_down', 
'pass_weight_mu_isolation_up', 'pass_weight_mu_isolation_down', 'pass_weight_mu_id_up', 'pass_weight_mu_id_down',
'pass_weight_mu_trigger_iso_up', 'pass_weight_mu_trigger_iso_down', 'pass_weight_mu_trigger_noniso_up', 'pass_weight_mu_trigger_noniso_down', 
'pass_weight_PSFSR_DYJets_up', 'pass_weight_PSFSR_DYJets_down', 'pass_weight_PSISR_DYJets_up', 'pass_weight_PSISR_DYJets_down',
'pass_weight_PSFSR_WH_up', 'pass_weight_PSFSR_WH_down', 'pass_weight_PSISR_WH_up', 'pass_weight_PSISR_WH_down', 
'pass_weight_PSFSR_ZH_up', 'pass_weight_PSFSR_ZH_down', 'pass_weight_PSISR_ZH_up', 'pass_weight_PSISR_ZH_down',
'pass_weight_PSFSR_VBF_up', 'pass_weight_PSFSR_VBF_down', 'pass_weight_PSISR_VBF_up', 'pass_weight_PSISR_VBF_down',
'pass_weight_PSFSR_WJetsLNu_up', 'pass_weight_PSFSR_WJetsLNu_down', 'pass_weight_PSISR_WJetsLNu_up', 'pass_weight_PSISR_WJetsLNu_down', 
'pass_weight_PSFSR_ttH_up', 'pass_weight_PSFSR_ttH_down','pass_weight_PSISR_ttH_up', 'pass_weight_PSISR_ttH_down',
'pass_weight_PSFSR_ggF_up', 'pass_weight_PSFSR_ggF_down', 'pass_weight_PSISR_ggF_up', 'pass_weight_PSISR_ggF_down',

'pass_weight_qcd_scale_TTbar_up', 'pass_weight_qcd_scale_TTbar_down',
'pass_weight_pdf_acceptance_TTbar_up', 'pass_weight_pdf_acceptance_TTbar_down', 
'pass_weight_PSFSR_TTbar_up', 'pass_weight_PSFSR_TTbar_down', 'pass_weight_PSISR_TTbar_up', 'pass_weight_PSISR_TTbar_down', 

'pass_weight_pdf_acceptance_SingleTop_up', 'pass_weight_pdf_acceptance_SingleTop_down',
'pass_weight_qcd_scale_SingleTop_up', 'pass_weight_qcd_scale_SingleTop_down',
'pass_weight_PSFSR_SingleTop_up', 'pass_weight_PSFSR_SingleTop_down', 'pass_weight_PSISR_SingleTop_up', 'pass_weight_PSISR_SingleTop_down',

'pass_top_reweighting_up', 'pass_top_reweighting_down',
'pass_UES_up', 'pass_UES_down',

'pass_JES_FlavorQCD_up', 'pass_JES_FlavorQCD_down', 'pass_JES_RelativeBal_up', 'pass_JES_RelativeBal_down', 
'pass_JES_HF_up', 'pass_JES_HF_down', 'pass_JES_BBEC1_up', 'pass_JES_BBEC1_down', 'pass_JES_EC2_up', 'pass_JES_EC2_down',
'pass_JES_Absolute_up', 'pass_JES_Absolute_down', 'pass_JES_BBEC1_year_up', 'pass_JES_BBEC1_year_down', 'pass_JES_RelativeSample_year_up',
'pass_JES_RelativeSample_year_down', 'pass_JES_EC2_year_up', 'pass_JES_EC2_year_down', 'pass_JES_HF_year_up', 'pass_JES_HF_year_down', 
'pass_JES_Absolute_year_up', 'pass_JES_Absolute_year_down', 'pass_JER_year_up', 'pass_JER_year_down',

#add this for Vtagger
'pass_weight_Vtagger_up',
'pass_weight_Vtagger_down',
'pass_weight_JMS_up',
'pass_weight_JMS_down',
'pass_weight_JMR_up',
'pass_weight_JMR_down',


]
    return systematicsList_pass