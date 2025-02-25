#bkgs = ["TTbar", "WJetsLNu", "SingleTop", "DYJets", "WQQ", "ZQQ", "Diboson", 'EWKWjets', 'EWKZjets']
bkgs = ["TTbar", "WJetsLNu", "SingleTop", "DYJets", "WQQ", "ZQQ", "Diboson", 'EWKWjets']


sigs = ["ggF", "VBF", "WH", "ZH", "ttH"]


#samples = sigs + bkgs + ["Fake"]
#samples = sigs + bkgs 


def get_systematic_dict(years):
    """
    The following dictionaries have the following convention,
        key [str] --> name of systematic to store in the histogram / template
        value [tuple] --> (t1, t2, t3):
            t1 [list]: years to process the up/down variations for (store nominal value for the other years)
            t2 [list]: samples to apply systematic for (store nominal value for the other samples)
            t3 [dict]:
                key(s): the channels to apply the systematic for (store nominal value for the other channels)
                value(s): the name of the variable in the parquet for that channel
    """

    radiation = {      # ISR/FSR
       "fail_weight_PSFSR": (  years,  sigs + ["TTbar", "WJetsLNu", "SingleTop"],
            {"ele": "weight_ele_PSFSR", "mu": "weight_mu_PSFSR"},
        ),
        "fail_weight_PSISR": (  years,  sigs + ["TTbar", "WJetsLNu", "SingleTop"],
            {"ele": "weight_ele_PSISR", "mu": "weight_mu_PSISR"},
        ),       
   }

    
    COMMON_systs_correlated = {
        "fail_weight_pileup_id": (
            years,
            sigs + bkgs,
            {"ele": "weight_ele_pileupIDSF", "mu": "weight_mu_pileupIDSF"},
        ),
   
       
        # systematics applied only on WJets & DYJets
        "fail_weight_d1K_NLO": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_d1K_NLO", "mu": "weight_mu_d1K_NLO"},
        ),
        "fail_weight_d2K_NLO": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_d2K_NLO", "mu": "weight_mu_d2K_NLO"},
        ),
        "fail_weight_d3K_NLO": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_d3K_NLO", "mu": "weight_mu_d3K_NLO"},
        ),
        "fail_weight_d1kappa_EW": (
            years,
            ["WJetsLNu", "DYJets"],
            {"ele": "weight_ele_d1kappa_EW", "mu": "weight_mu_d1kappa_EW"},
        ),
        "fail_weight_W_d2kappa_EW": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_W_d2kappa_EW", "mu": "weight_mu_W_d2kappa_EW"},
        ),
        "fail_weight_W_d3kappa_EW": (
            years,
            ["WJetsLNu"],
            {"ele": "weight_ele_W_d3kappa_EW", "mu": "weight_mu_W_d3kappa_EW"},
        ),
        "fail_weight_Z_d2kappa_EW": (
            years,
            ["DYJets"],
            {"ele": "weight_ele_Z_d2kappa_EW", "mu": "weight_mu_Z_d2kappa_EW"},
        ),
        "fail_weight_Z_d3kappa_EW": (
            years,
            ["DYJets"],
            {"ele": "weight_ele_Z_d3kappa_EW", "mu": "weight_mu_Z_d3kappa_EW"},
        ),
        # systematics for electron channel
        "fail_weight_ele_id": (
            years,
            sigs + bkgs,
            {"ele": "weight_ele_id_electron"},
        ),
        "fail_weight_ele_reco": (
            years,
            sigs + bkgs,
            {"ele": "weight_ele_reco_electron"},
        ),
        # systematics for muon channel
        "fail_weight_mu_isolation": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_isolation_muon"},
        ),
        "fail_weight_mu_id": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_id_muon"},
        ),
        "fail_weight_mu_trigger_iso": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_trigger_iso_muon"},
        ),
        "fail_weight_mu_trigger_noniso": (
            years,
            sigs + bkgs,
            {"mu": "weight_mu_trigger_noniso_muon"},
        ),
    }

    COMMON_systs_uncorrelated = {}
    for year in years:
        COMMON_systs_uncorrelated = {
            **COMMON_systs_uncorrelated,
            **{
                f"fail_weight_pileup": (
                    [year],
                    sigs + bkgs,
                    {"ele": "weight_ele_pileup", "mu": "weight_mu_pileup"},
                ),
            },
        }
        if year != "2018":
            COMMON_systs_uncorrelated = {
                **COMMON_systs_uncorrelated,
                **{
                    f"fail_weight_L1Prefiring": (
                        [year],
                        sigs + bkgs,
                        {"ele": "weight_ele_L1Prefiring", "mu": "weight_mu_L1Prefiring"},
                    ),
                },
            }

    # btag syst. have a different treatment because they are not stored in the nominal
    BTAG_systs_correlated = {
        "fail_weight_btagSFlightCorrelated": (
            years,
            sigs + bkgs,
            {"ele": "weight_btagSFlightCorrelated", "mu": "weight_btagSFlightCorrelated"},
        ),
        "fail_weight_btagSFbcCorrelated": (
            years,
            sigs + bkgs,
            {"ele": "weight_btagSFbcCorrelated", "mu": "weight_btagSFbcCorrelated"},
        ),
    }

    BTAG_systs_uncorrelated = {}
    for year in years:
        if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
            yearlabel = "2016"
        else:
            yearlabel = year

        BTAG_systs_uncorrelated = {
            **BTAG_systs_uncorrelated,
            **{
                f"fail_weight_btagSFlight": (
                    year,
                    sigs + bkgs,
                    {"ele": f"weight_btagSFlight{yearlabel}", "mu": f"weight_btagSFlight{yearlabel}"},
                ),
                f"fail_weight_btagSFbc": (
                    year,
                    sigs + bkgs,
                    {"ele": f"weight_btagSFbc{yearlabel}", "mu": f"weight_btagSFbc{yearlabel}"},
                ),
            },
        }

    # JEC / UES
    JEC_systs_correlated = {
        "fail_JES_FlavorQCD": (
            years,
            sigs + bkgs,
            {"ele": "JES_FlavorQCD", "mu": "JES_FlavorQCD"},
        ),
        "fail_JES_RelativeBal": (
            years,
            sigs + bkgs,
            {"ele": "JES_RelativeBal", "mu": "JES_RelativeBal"},
        ),
        "fail_JES_HF": (
            years,
            sigs + bkgs,
            {"ele": "JES_HF", "mu": "JES_HF"},
        ),
        "fail_JES_BBEC1": (
            years,
            sigs + bkgs,
            {"ele": "JES_BBEC1", "mu": "JES_BBEC1"},
        ),
        "fail_JES_EC2": (
            years,
            sigs + bkgs,
            {"ele": "JES_EC2", "mu": "JES_EC2"},
        ),
        "fail_JES_Absolute": (
            years,
            sigs + bkgs,
            {"ele": "JES_Absolute", "mu": "JES_Absolute"},
        ),
    }

    JEC_systs_uncorrelated = {}
    for year in years:
        if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
            yearlabel = "2016"
        else:
            yearlabel = year

        JEC_systs_uncorrelated = {
            **JEC_systs_uncorrelated,
            **{
                "fail_JES_BBEC1_year": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_BBEC1_{yearlabel}", "mu": f"JES_BBEC1_{yearlabel}"},
                ),
                "fail_JES_RelativeSample_year": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_RelativeSample_{yearlabel}", "mu": f"JES_RelativeSample_{yearlabel}"},
                ),
                "fail_JES_EC2_year": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_EC2_{yearlabel}", "mu": f"JES_EC2_{yearlabel}"},
                ),
                "fail_JES_HF_year": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_HF_{yearlabel}", "mu": f"JES_HF_{yearlabel}"},
                ),
                "fail_JES_Absolute_year": (
                    year,
                    sigs + bkgs,
                    {"ele": f"JES_Absolute_{yearlabel}", "mu": f"JES_Absolute_{yearlabel}"},
                ),
               "fail_JER_year": (
                    year,
                    sigs + bkgs,
                    {"ele": "JER", "mu": "JER"},
                ),
            },
        }


    

    UES_systs = {}
    for year in years:
        if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
            yearlabel = "2016"
        else:
            yearlabel = year

        UES_systs = {
            **UES_systs,
            **{
        "fail_UES": (
            years,
            sigs + bkgs,
            {"ele": "UES", "mu": "UES"},
        ),
        },
    }

    TRIGGER_systs = {}
    for year in years:
        if "APV" in year:  # all APV parquets don't have APV explicitly in the systematics
            yearlabel = "2016"
        else:
            yearlabel = year

        TRIGGER_systs = {
            **TRIGGER_systs,
            **{
        "fail_weight_trigger": (
            years,
            sigs + bkgs,
            {"ele": "weight_ele_trigger", },
        ),
        },
    }

    SYST_DICT = {
        "psrad": {**radiation},
        "common": {**COMMON_systs_correlated, **COMMON_systs_uncorrelated},
        "btag1": {**BTAG_systs_correlated},
        "btag2": {**BTAG_systs_uncorrelated},
        "JEC": {**JEC_systs_correlated, **JEC_systs_uncorrelated},

        "UES_systs": {**UES_systs, **UES_systs, },
        "TRIGGER_systs": {**TRIGGER_systs, **TRIGGER_systs},
    }

    return SYST_DICT
