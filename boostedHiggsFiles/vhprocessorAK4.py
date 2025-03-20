import importlib.resources
import json
import logging
import os
import pathlib
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents.methods import candidate
from boostedhiggs.utils import ELE_PDGID, MU_PDGID, match_V

from coffea.nanoevents.methods.nanoaod import GenParticleArray, JetArray

logger = logging.getLogger(__name__)

from boostedhiggs.corrections import (
    add_HiggsEW_kFactors,
    add_lepton_weight,
    add_pileup_weight,
    add_pileupid_weights,
    add_ps_weight,
    add_VJets_kFactors,
    btagWPs,
    corrected_msoftdrop,
    get_btag_weights,
    get_JetVetoMap,
    get_jec_jets,

    get_jec_jets2,
    get_jmsr,
    getJECVariables,
    getJECVariables_Higgs,
    getJMSRVariables,
    met_factory,
    add_TopPtReweighting,
)
from boostedhiggs.utils import VScore, match_H2, match_Top, match_V, sigs, get_pid_mask

from .run_tagger_inference import runInferenceTriton

from .SkimmerABC import *

warnings.filterwarnings("ignore", message="Found duplicate branch ")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Missing cross-reference index ")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.seterr(invalid="ignore")


#**************************************************

def build_p4(cand):
    return ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
            "charge": cand.charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

def VScore(goodFatJetsSelected):
    num = ( goodFatJetsSelected.particleNetMD_Xbb + goodFatJetsSelected.particleNetMD_Xcc + goodFatJetsSelected.particleNetMD_Xqq )
    den = ( goodFatJetsSelected.particleNetMD_Xbb + goodFatJetsSelected.particleNetMD_Xcc + goodFatJetsSelected.particleNetMD_Xqq + goodFatJetsSelected.particleNetMD_QCD
    )
    score = num / den
    return score

def getNumberBTaggedJets(ak4jets, higgsJet, VJet,year):
    dr_ak8Jets_HiggsCandidateJet_Uncertainty = ak4jets.delta_r(higgsJet)
    dr_ak8Jets_VCandidateJet_Uncertainty = ak4jets.delta_r(VJet)
    ak4_outsideBothJets_Uncertainty = ak4jets[ (dr_ak8Jets_HiggsCandidateJet_Uncertainty > 0.8) & (dr_ak8Jets_VCandidateJet_Uncertainty  > 0.8) ]
    #NumOtherJetsOutsideBothJets_Uncertainty_down = ak.num(ak4_outsideBothJets_Uncertainty_down)
    NumOtherJetsOutsideBothJets_Uncertainty = ak.sum( ak4_outsideBothJets_Uncertainty.btagDeepFlavB > btagWPs["deepJet"][year]["M"],axis=1, )
    return NumOtherJetsOutsideBothJets_Uncertainty

class vhprocessorAK4(processor.ProcessorABC):
    def __init__(
        self,
        year="2017",
        yearmod="",
        channels=["ele", "mu"],
        output_location="./outfiles/",
        inference=False,
        systematics=False,
        getLPweights=False,
        uselooselep=False,
    ):
        self._year = year
        self._yearmod = yearmod
        self._channels = channels
        self._systematics = systematics
        self._getLPweights = getLPweights
        self._uselooselep = uselooselep
        #self._fakevalidation = fakevalidation
        self._output_location = output_location
        # trigger paths
        with importlib.resources.path("boostedhiggs.data", "triggers.json") as path:
            with open(path, "r") as f:
                self._HLTs = json.load(f)[self._year]

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with importlib.resources.path("boostedhiggs.data", "metfilters.json") as path:
            with open(path, "r") as f:
                self._metfilters = json.load(f)[self._year]

        if self._year == "2018":
            self.dataset_per_ch = {
                "ele": "EGamma",
                "mu": "SingleMuon",
            }
        else:
            self.dataset_per_ch = {
                "ele": "SingleElectron",
                "mu": "SingleMuon",
            }

        self.jecs = {
            "JES": "JES_jes",
            "JER": "JER",
            # individual sources
            "JES_FlavorQCD": "JES_FlavorQCD",
            "JES_RelativeBal": "JES_RelativeBal",
            "JES_HF": "JES_HF",
            "JES_BBEC1": "JES_BBEC1",
            "JES_EC2": "JES_EC2",
            "JES_Absolute": "JES_Absolute",
            f"JES_BBEC1_{self._year}": f"JES_BBEC1_{self._year}",
            f"JES_RelativeSample_{self._year}": f"JES_RelativeSample_{self._year}",
            f"JES_EC2_{self._year}": f"JES_EC2_{self._year}",
            f"JES_HF_{self._year}": f"JES_HF_{self._year}",
            f"JES_Absolute_{self._year}": f"JES_Absolute_{self._year}",
            "JES_Total": "JES_Total",
        }

        # for tagger inference
        self._inference = inference
        self.tagger_resources_path = str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"

    @property
    def accumulator(self):
        return self._accumulator

    def save_dfs_parquet(self, fname, dfs_dict, ch):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries
                pq.write_table(table, self._output_location + ch + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def add_selection(self, name: str, sel: np.ndarray, channel: str = "all"):
        """Adds selection to PackedSelection object and the cutflow dictionary"""
        channels = self._channels if channel == "all" else [channel]

        for ch in channels:
            if ch not in self._channels:
                logger.warning(f"Attempted to add selection to unexpected channel: {ch} not in %s" % (self._channels))
                continue

            # add selection
            self.selections[ch].add(name, sel)
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            if self.isMC:
                weight = self.weights[ch].partial_weight(["genweight"])  
                #print('weight', weight)
                self.cutflows[ch][name] = float(weight[selection_ch].sum())
            else:
                self.cutflows[ch][name] = np.sum(selection_ch)

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts and with the branches listed in self._skimvars"""

        dataset = events.metadata["dataset"]
        self.isMC = hasattr(events, "genWeight")
        self.isSignal = True if ("HToWW" in dataset) or ("ttHToNonbb" in dataset) else False
        nevents = len(events)
        self.weights = {ch: Weights(nevents, storeIndividual=True) for ch in self._channels}
        self.selections = {ch: PackedSelection() for ch in self._channels}
        self.cutflows = {ch: {} for ch in self._channels}

        if "TT" in dataset or "ST_" in dataset or "DYJets" in dataset:
            sumgenweight = ak.sum(np.sign(events.genWeight)) if self.isMC else nevents
        else:
            sumgenweight = ak.sum(events.genWeight) if self.isMC else nevents

        sumpdfweight = {}
        sumlheweight = {}

        if "TT" in dataset or "ST_" in dataset:
            if "LHEScaleWeight" in events.fields and self.isMC:
                if len(events.LHEScaleWeight[0]) == 9:
                    for i in range(len(events.LHEScaleWeight[0])):
                        sumlheweight[i] = ak.sum(events.LHEScaleWeight[:, i] * np.sign(events.genWeight))
            if "LHEPdfWeight" in events.fields and self.isMC:
                for i in range(len(events.LHEPdfWeight[0])):
                    sumpdfweight[i] = ak.sum(events.LHEPdfWeight[:, i] * np.sign(events.genWeight))
        else:
            if "LHEScaleWeight" in events.fields and self.isMC:
                if len(events.LHEScaleWeight[0]) == 9:
                    for i in range(len(events.LHEScaleWeight[0])):
                        sumlheweight[i] = ak.sum(events.LHEScaleWeight[:, i] * events.genWeight)
            if "LHEPdfWeight" in events.fields and self.isMC:
                for i in range(len(events.LHEPdfWeight[0])):
                    sumpdfweight[i] = ak.sum(events.LHEPdfWeight[:, i] * events.genWeight)

        # add genweight before filling cutflow
        if self.isMC:
            for ch in self._channels:
                #if "TT" in dataset or "ST_" in dataset:
                if "TT" in dataset or "ST_" in dataset or "DYJets" in dataset:
                    self.weights[ch].add("genweight", np.sign(events.genWeight))
                else:
                    self.weights[ch].add("genweight", events.genWeight)


        ######################
        # Trigger
        ######################

        trigger = {}
        for ch in ["ele", "mu_lowpt", "mu_highpt"]:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            for t in self._HLTs[ch]:
                if t in events.HLT.fields:
                    trigger[ch] = trigger[ch] | events.HLT[t]

        trigger["ele"] = trigger["ele"] & (~trigger["mu_lowpt"]) & (~trigger["mu_highpt"])
        trigger["mu_highpt"] = trigger["mu_highpt"] & (~trigger["ele"])
        trigger["mu_lowpt"] = trigger["mu_lowpt"] & (~trigger["ele"])

        ######################
        # METFLITERS
        ######################

        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        ######################
        # OBJECT DEFINITION
        ######################

        # OBJECT: muons
        muons = ak.with_field(events.Muon, 0, "flavor")

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4) & muons.mediumId
            & (((muons.pfRelIso04_all < 0.20) & (muons.pt < 55)) | (muons.pt >= 55) & (muons.miniPFRelIso_all < 0.2))
            # additional cuts
            & (np.abs(muons.dz) < 0.1) & (np.abs(muons.dxy) < 0.02)
        )

        n_good_muons = ak.sum(good_muons, axis=1)

        # OBJECT: electrons
        electrons = ak.with_field(events.Electron, 1, "flavor")

        good_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.5)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (electrons.mvaFall17V2noIso_WP90)
            & (((electrons.pfRelIso03_all < 0.15) & (electrons.pt < 120)) | (electrons.pt >= 120))
            # additional cuts
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
        )

        n_good_electrons = ak.sum(good_electrons, axis=1)

        # OBJECT: candidate lepton
        goodleptons = ak.concatenate([muons[good_muons], electrons[good_electrons]], axis=1)  # concat muons and electrons
        goodleptons = goodleptons[ak.argsort(goodleptons.pt, ascending=False)]  # sort by pt

        candidatelep = ak.firsts(goodleptons)  # pick highest pt
        candidatelep_p4 = build_p4(candidatelep)  # build p4 for candidate lepton

        lep_reliso = (
            candidatelep.pfRelIso04_all if hasattr(candidatelep, "pfRelIso04_all") else candidatelep.pfRelIso03_all
        )  # reliso for candidate lepton
        lep_miso = candidatelep.miniPFRelIso_all  # miniso for candidate lepton

        ngood_leptons = ak.num(goodleptons, axis=1)

        # OBJECT: AK8 fatjets
        fatjets = events.FatJet
        fatjets["msdcorr"] = corrected_msoftdrop(fatjets)
        fatjet_selector = (fatjets.pt > 200) & (abs(fatjets.eta) < 2.5) & fatjets.isTight
        good_fatjets = fatjets[fatjet_selector]
        good_fatjets = good_fatjets[ak.argsort(good_fatjets.pt, ascending=False)]  # sort them by pt
        NumFatjets = ak.num(good_fatjets)

        #JETS**************************************************
        #this applies JEC to all the fat jets (same as farouk)
        good_fatjets, jec_shifted_fatjetvars = get_jec_jets(
            events, good_fatjets, self._year, not self.isMC, self.jecs, fatjets=True
        )
        #******************************************************

        # OBJECT: candidate fatjet
        fj_idx_lep = ak.argmin(good_fatjets.delta_r(candidatelep_p4), axis=1, keepdims=True)
        candidatefj = ak.firsts(good_fatjets[fj_idx_lep])
        lep_fj_dr = candidatefj.delta_r(candidatelep_p4)

        #jmsr_shifted_fatjetvars = get_jmsr(good_fatjets[fj_idx_lep], num_jets=1, year=self._year, isData=not self.isMC)
        #*************************************************************************
        # VH jet   /differs from HWW processor, but Farouks added this into hww processor now
        deltaR_lepton_all_jets = candidatelep_p4.delta_r(good_fatjets)
        minDeltaR = ak.argmin(deltaR_lepton_all_jets, axis=1)
        fatJetIndices = ak.local_index(good_fatjets, axis=1)
        mask1 = fatJetIndices != minDeltaR
        allScores = VScore(good_fatjets)
        masked = allScores[mask1]
        secondFJ = good_fatjets[allScores == ak.max(masked, axis=1)]
        second_fj = ak.firsts(secondFJ)
        VCandidateVScore = VScore(second_fj)
        VCandidate_Mass = second_fj.msdcorr

        #get index of the V
        maxScoreIndexMask = (allScores == ak.max(masked, axis=1))
        #VIndex = ak.firsts(fatJetIndices[maxScoreIndexMask])
        VIndex = fatJetIndices[maxScoreIndexMask]

        dr_two_jets = candidatefj.delta_r(second_fj)

        #jmsr_shifted_fatjetvars = get_jmsr(secondFJ, num_jets=1, year=self._year, isData=not self.isMC)

        #jmsr_shifted_fatjetvars = get_jmsr(secondFJ, num_jets=1, year=self._year, isData=not self.isMC)
        #V_nom = jmsr_shifted_fatjetvars["msoftdrop"][""]

        #print('correctedMass', correctedVbosonNominalMass)
        #*************************************************************************
        # OBJECT: AK4 jets
        jets, jec_shifted_jetvars = get_jec_jets(events, events.Jet, self._year, not self.isMC, self.jecs, fatjets=False)
        met = met_factory.build(events.MET, jets, {}) if self.isMC else events.MET

        #jet_selector = ( (jets.pt > 30) & (abs(jets.eta) < 5.0) & jets.isTight & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2)) )
        jet_selector =  (
            (jets.pt > 15)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2))
            & (jets.chEmEF + jets.neEmEF < 0.9)  # neutral and charged energy fraction
        )
        jets = jets[jet_selector]
        jet_veto_map, cut_jetveto = get_JetVetoMap(jets, self._year)
        final_mask = (jets.pt > 30) & jet_veto_map
        jets = jets[final_mask]

        #goodjets = jets[jet_selector]
        bjet_selector = (jets.delta_r(candidatefj) > 0.8) & (jets.delta_r(second_fj) > 0.8) & (abs(jets.eta) < 2.5)
        dr_ak8Jets_HiggsCandidateJet = jets.delta_r(candidatefj)
        dr_ak8Jets_VCandidateJet = jets.delta_r(second_fj)
        ak4_outsideBothJets = jets[ (dr_ak8Jets_HiggsCandidateJet > 0.8) & (dr_ak8Jets_VCandidateJet  > 0.8) ]
        NumOtherJetsOutsideBothJets = ak.num(ak4_outsideBothJets)
        n_bjets_M_OutsideBothJets = ak.sum( ak4_outsideBothJets.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"],axis=1, )

        #bjet_selector = (jet_selector) & (jets.delta_r(candidatefj) > 0.8) & (jets.delta_r(second_fj) > 0.8) & (abs(jets.eta) < 2.5)
        #dr_ak8Jets_HiggsCandidateJet = goodjets.delta_r(candidatefj)
        #dr_ak8Jets_VCandidateJet = goodjets.delta_r(second_fj)
        #ak4_outsideBothJets = goodjets[ (dr_ak8Jets_HiggsCandidateJet > 0.8) & (dr_ak8Jets_VCandidateJet  > 0.8) ]
        #NumOtherJetsOutsideBothJets = ak.num(ak4_outsideBothJets)
        #n_bjets_M_OutsideBothJets = ak.sum( ak4_outsideBothJets.btagDeepFlavB > btagWPs["deepJet"][self._year]["M"],axis=1, )

#farouk's stuff for potential veto
        ak4_outside_ak8_selector = jets.delta_r(candidatefj) > 0.8
        ak4_outside_ak8 = jets[ak4_outside_ak8_selector]
        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]
        deta = abs(ak.firsts(jet1).eta - ak.firsts(jet2).eta)
        mjj = (ak.firsts(jet1) + ak.firsts(jet2)).mass
#*****************************************************************

        jec_shifted_jetvars_filtered = { var: {shift: values[final_mask] for shift, values in shifts.items()} for var, shifts in jec_shifted_jetvars.items()
}

    #for ak4 jet uncertaintites only
    #Jet_nominal = jec_shifted_jetvars['pt']['']
        JES_down = jec_shifted_jetvars_filtered['pt']['JES_down']
        JES_up = jec_shifted_jetvars_filtered['pt']['JES_up']
        JER_down = jec_shifted_jetvars_filtered['pt']['JER_down']
        JER_up = jec_shifted_jetvars_filtered['pt']['JER_up']
        JES_FlavorQCD_up = jec_shifted_jetvars_filtered['pt']['JES_FlavorQCD_up']
        JES_FlavorQCD_down = jec_shifted_jetvars_filtered['pt']['JES_FlavorQCD_down']
        JES_RelativeBal_up = jec_shifted_jetvars_filtered['pt']['JES_RelativeBal_up']
        JES_RelativeBal_down = jec_shifted_jetvars_filtered['pt']['JES_RelativeBal_down']
        JES_HF_up = jec_shifted_jetvars_filtered['pt']['JES_HF_up']
        JES_HF_down = jec_shifted_jetvars_filtered['pt']['JES_HF_down']
        JES_BBEC1_up = jec_shifted_jetvars_filtered['pt']['JES_BBEC1_up']
        JES_BBEC1_down = jec_shifted_jetvars_filtered['pt']['JES_BBEC1_down']
        JES_EC2_up = jec_shifted_jetvars_filtered['pt']['JES_EC2_up']
        JES_EC2_down = jec_shifted_jetvars_filtered['pt']['JES_EC2_down']
        JES_Absolute_up = jec_shifted_jetvars_filtered['pt']['JES_Absolute_up']
        JES_Absolute_down = jec_shifted_jetvars_filtered['pt']['JES_Absolute_down']
        
        JES_BBEC1_year_up = jec_shifted_jetvars_filtered['pt'][f'JES_BBEC1_{self._year}_up']
        JES_BBEC1_year_down = jec_shifted_jetvars_filtered['pt'][f'JES_BBEC1_{self._year}_down']
        JES_RelativeSample_year_up = jec_shifted_jetvars_filtered['pt'][f'JES_RelativeSample_{self._year}_up']
        JES_RelativeSample_year_down = jec_shifted_jetvars_filtered['pt'][f'JES_RelativeSample_{self._year}_down']
        JES_EC2_year_up = jec_shifted_jetvars_filtered['pt'][f'JES_EC2_{self._year}_up']
        JES_EC2_year_down = jec_shifted_jetvars_filtered['pt'][f'JES_EC2_{self._year}_down']
        JES_HF_year_up = jec_shifted_jetvars_filtered['pt'][f'JES_HF_{self._year}_up']
        JES_HF_year_down = jec_shifted_jetvars_filtered['pt'][f'JES_HF_{self._year}_down']
        JES_Absolute_year_up = jec_shifted_jetvars_filtered['pt'][f'JES_Absolute_{self._year}_up']
        JES_Absolute_year_down = jec_shifted_jetvars_filtered['pt'][f'JES_Absolute_{self._year}_down']
        
        JES_Total_up = jec_shifted_jetvars_filtered['pt']['JES_Total_up']
        JES_Total_down = jec_shifted_jetvars_filtered['pt']['JES_Total_down']


        #original code
        #jet_selector_JES_down = ( (JES_down > 30) & (abs(jets.eta) < 5.0)  & jets.isTight & ((jets.pt >= 50) | ((jets.pt < 50) & (jets.puId & 2) == 2)))
        jet_selector_JES_down = JES_down > 30
        goodjets_JES_down = jets[jet_selector_JES_down]
        numBJets_JES_down =getNumberBTaggedJets(goodjets_JES_down, candidatefj, second_fj,self._year)
        jet_selector_JES_up = ( (JES_up > 30) )
        goodjets_JES_up = jets[jet_selector_JES_up]
        numBJets_JES_up =getNumberBTaggedJets(goodjets_JES_up, candidatefj, second_fj,self._year)

        jet_selector_JER_down = JER_down > 30
        goodjets_JER_down = jets[jet_selector_JER_down]
        numBJets_JER_down =getNumberBTaggedJets(goodjets_JER_down, candidatefj, second_fj,self._year)
        jet_selector_JER_up = JER_up > 30
        goodjets_JER_up = jets[jet_selector_JER_up]
        numBJets_JER_up =getNumberBTaggedJets(goodjets_JER_up, candidatefj, second_fj,self._year)

        jet_selector_JES_FlavorQCD_up = JES_FlavorQCD_up > 30
        goodjets_JES_FlavorQCD_up = jets[jet_selector_JES_FlavorQCD_up]
        numBJets_JES_FlavorQCD_up =getNumberBTaggedJets(goodjets_JES_FlavorQCD_up, candidatefj, second_fj,self._year)
        jet_selector_JES_FlavorQCD_down = JES_FlavorQCD_down > 30
        goodjets_JES_FlavorQCD_down = jets[jet_selector_JES_FlavorQCD_down]
        numBJets_JES_FlavorQCD_down =getNumberBTaggedJets(goodjets_JES_FlavorQCD_down, candidatefj, second_fj,self._year)

        jet_selector_JES_RelativeBal_up = JES_RelativeBal_up > 30
        goodjets_JES_RelativeBal_up = jets[jet_selector_JES_RelativeBal_up]
        numBJets_JES_RelativeBal_up =getNumberBTaggedJets(goodjets_JES_RelativeBal_up, candidatefj, second_fj,self._year)

        jet_selector_JES_RelativeBal_down = JES_RelativeBal_down > 30
        goodjets_JES_RelativeBal_down = jets[jet_selector_JES_RelativeBal_down]
        numBJets_JES_RelativeBal_down =getNumberBTaggedJets(goodjets_JES_RelativeBal_down, candidatefj, second_fj,self._year)

        jet_selector_JES_HF_up = JES_HF_up > 30
        goodjets_JES_HF_up = jets[jet_selector_JES_HF_up]
        numBJets_JES_HF_up =getNumberBTaggedJets(goodjets_JES_HF_up, candidatefj, second_fj,self._year)

        jet_selector_JES_HF_down = JES_HF_down > 30
        goodjets_JES_HF_down = jets[jet_selector_JES_HF_down]
        numBJets_JES_HF_down =getNumberBTaggedJets(goodjets_JES_HF_down, candidatefj, second_fj,self._year)

        jet_selector_JES_BBEC1_up = JES_BBEC1_up > 30
        goodjets_JES_BBEC1_up = jets[jet_selector_JES_BBEC1_up]
        numBJets_JES_BBEC1_up =getNumberBTaggedJets(goodjets_JES_BBEC1_up, candidatefj, second_fj,self._year)

        jet_selector_JES_BBEC1_down = JES_BBEC1_down > 30
        goodjets_JES_BBEC1_down = jets[jet_selector_JES_BBEC1_down]
        numBJets_JES_BBEC1_down =getNumberBTaggedJets(goodjets_JES_BBEC1_down, candidatefj, second_fj,self._year)

        jet_selector_JES_EC2_up = JES_EC2_up > 30
        goodjets_JES_EC2_up = jets[jet_selector_JES_EC2_up]
        numBJets_JES_EC2_up =getNumberBTaggedJets(goodjets_JES_EC2_up, candidatefj, second_fj,self._year)

        jet_selector_JES_EC2_down = JES_EC2_down > 30
        goodjets_JES_EC2_down = jets[jet_selector_JES_EC2_down]
        numBJets_JES_EC2_down =getNumberBTaggedJets(goodjets_JES_EC2_down, candidatefj, second_fj,self._year)

        jet_selector_JES_Absolute_up = JES_Absolute_up > 30
        goodjets_JES_Absolute_up = jets[jet_selector_JES_Absolute_up]
        numBJets_JES_Absolute_up =getNumberBTaggedJets(goodjets_JES_Absolute_up, candidatefj, second_fj,self._year)

        jet_selector_JES_Absolute_down = JES_Absolute_down > 30
        goodjets_JES_Absolute_down = jets[jet_selector_JES_Absolute_down]
        numBJets_JES_Absolute_down =getNumberBTaggedJets(goodjets_JES_Absolute_down, candidatefj, second_fj,self._year)

        jet_selector_JES_BBEC1_year_up = JES_BBEC1_year_up > 30
        goodjets_JES_BBEC1_year_up = jets[jet_selector_JES_BBEC1_year_up]
        numBJets_JES_BBEC1_year_up =getNumberBTaggedJets(goodjets_JES_BBEC1_year_up, candidatefj, second_fj,self._year)

        jet_selector_JES_BBEC1_year_down = JES_BBEC1_year_down > 30
        goodjets_JES_BBEC1_year_down = jets[jet_selector_JES_BBEC1_year_down]
        numBJets_JES_BBEC1_year_down =getNumberBTaggedJets(goodjets_JES_BBEC1_year_down, candidatefj, second_fj,self._year)

        jet_selector_JES_RelativeSample_year_up = JES_RelativeSample_year_up > 30
        goodjets_JES_RelativeSample_year_up = jets[jet_selector_JES_RelativeSample_year_up]
        numBJets_JES_RelativeSample_year_up =getNumberBTaggedJets(goodjets_JES_RelativeSample_year_up, candidatefj, second_fj,self._year)

        jet_selector_JES_RelativeSample_year_down = JES_RelativeSample_year_down > 30
        goodjets_JES_RelativeSample_year_down = jets[jet_selector_JES_RelativeSample_year_down]
        numBJets_JES_RelativeSample_year_down =getNumberBTaggedJets(goodjets_JES_RelativeSample_year_down, candidatefj, second_fj,self._year)

        jet_selector_JES_EC2_year_up = JES_EC2_year_up > 30
        goodjets_JES_EC2_year_up = jets[jet_selector_JES_EC2_year_up]
        numBJetsJES_EC2_year_up =getNumberBTaggedJets(goodjets_JES_EC2_year_up, candidatefj, second_fj,self._year)

        jet_selector_JES_EC2_year_down = JES_EC2_year_down > 30
        goodjets_JES_EC2_year_down = jets[jet_selector_JES_EC2_year_down]
        numBJetsJES_EC2_year_down =getNumberBTaggedJets(goodjets_JES_EC2_year_down, candidatefj, second_fj,self._year)

        jet_selector_JES_HF_year_up = JES_HF_year_up > 30
        goodjets_JES_HF_year_up = jets[jet_selector_JES_HF_year_up]
        numBJetsJES_HF_year_up =getNumberBTaggedJets(goodjets_JES_HF_year_up, candidatefj, second_fj,self._year)

        jet_selector_JES_HF_year_down = JES_HF_year_down > 30
        goodjets_JES_HF_year_down = jets[jet_selector_JES_HF_year_down]
        numBJetsJES_HF_year_down =getNumberBTaggedJets(goodjets_JES_HF_year_down, candidatefj, second_fj,self._year)

        jet_selector_JES_Absolute_year_up = JES_Absolute_year_up > 30
        goodjets_JES_Absolute_year_up = jets[jet_selector_JES_Absolute_year_up]
        numBJetsJES_Absolute_year_up =getNumberBTaggedJets(goodjets_JES_Absolute_year_up, candidatefj, second_fj,self._year)

        jet_selector_JES_Absolute_year_down = JES_Absolute_year_down > 30
        goodjets_JES_Absolute_year_down = jets[jet_selector_JES_Absolute_year_down]
        numBJetsJES_Absolute_year_down =getNumberBTaggedJets(goodjets_JES_Absolute_year_down, candidatefj, second_fj,self._year)

        jet_selector_JES_Total_up = JES_Total_up > 30
        goodjets_JES_Total_up = jets[jet_selector_JES_Total_up]
        numBJetsJES_Total_up =getNumberBTaggedJets(goodjets_JES_Total_up, candidatefj, second_fj,self._year)

        jet_selector_JES_Total_down = JES_Total_down > 30
        goodjets_JES_Total_down = jets[jet_selector_JES_Total_down]
        numBJetsJES_Total_down =getNumberBTaggedJets(goodjets_JES_Total_down, candidatefj, second_fj,self._year)


        #end ak4 jet uncertainties stuff*****************************************************************************

        mt_lep_met = np.sqrt( 2.0 * candidatelep_p4.pt * met.pt * (ak.ones_like(met.pt) - np.cos(candidatelep_p4.delta_phi(met))))
        met_fj_dphi = candidatefj.delta_phi(met)




        if self.isMC: 
            cutoff = 4.
            cutOnPU = np.ones(nevents,dtype='bool')
            pweights = corrections.get_pileup_weight_raghav(self._year, events.Pileup.nPU.to_numpy())
            pw_pass = ((pweights["nominal"] <= cutoff)*(pweights["up"] <= cutoff)*(pweights["down"] <= cutoff))
            #print('pw_pass', pw_pass)
        else:
            pw_pass = np.ones(nevents,dtype='bool')

#add in higgs mass for fun
        candidateHiggs = ak.zip (
        {
        "pt": candidatefj.pt,
        "eta": candidatefj.eta,
        "phi": candidatefj.phi,
        "mass": candidatefj.mass,
        },
            with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

        candidateNeutrino = ak.zip(
                {
                    "pt": met.pt,
                    "eta": candidateHiggs.eta,
                    "phi": met.phi,
                    "mass": 0,
                    "charge": 0,
                },
                with_name="PtEtaPhiMCandidate",behavior=candidate.behavior,)

        rec1 = candidatelep_p4 + candidateNeutrino
        rec2 = candidateHiggs - candidatelep_p4
        rec_higgs = rec1 + rec2

        if self.isMC: 
             genlep = events.GenPart[get_pid_mask(events.GenPart, [ELE_PDGID, MU_PDGID], byall=False)* events.GenPart.hasFlags(["fromHardProcess", "isLastCopy", "isPrompt"])]
             GenLep = ak.zip(         {
                "pt": genlep.pt,
                "eta": genlep.eta,
                "phi": genlep.phi,
                "mass": genlep.mass,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
             dR_genlep_recolep = GenLep.delta_r(candidatelep_p4)
             genlep_idx = ak.argmin(dR_genlep_recolep, axis=1, keepdims=True)
             dR_genlep_recolep = ak.firsts(dR_genlep_recolep[genlep_idx])



#add genmatching for V - use Farouk's functino here
        if self.isMC: 
            genVars, matched_mask = match_V(events.GenPart, second_fj )  #get gen Vars for matched V boson only

        if self.isMC:
             if self.isSignal:
                  HiggsgenVars = match_H2(events.GenPart, candidatefj, fatjet_pt=candidatefj.pt)
                  genVars = {**genVars, **HiggsgenVars}

        ######################
        # Store variables
        ######################

        variables = {
            "n_good_electrons": n_good_electrons, # n_good_electrons = ak.sum(good_electrons, axis=1)
            "n_good_muons": n_good_muons, #     n_good_muons = ak.sum(good_muons, axis=1)
            "lep_pt": candidatelep.pt,
            "lep_eta": candidatelep.eta,
            "lep_isolation": lep_reliso,
            "lep_misolation": lep_miso,
  
            "lep_fj_dr": lep_fj_dr, #  lep_fj_dr = candidatefj.delta_r(candidatelep_p4)
            "lep_met_mt": mt_lep_met, 
            "met_fj_dphi": met_fj_dphi,
            "met_pt": met.pt,
            "NumFatjets": NumFatjets, # NumFatjets = ak.num(good_fatjets)
            "h_fj_pt": candidatefj.pt, #Higgs
            "ReconVCandidateFatJetVScore": VCandidateVScore, # VCandidateVScore = VScore(second_fj)
            "ReconVCandidateMass": second_fj.msoftdrop,  #VCandidate_Mass = second_fj.msdcorr
            "numberAK4JetsOutsideFatJets": NumOtherJetsOutsideBothJets,
            "numberBJets_Medium_OutsideFatJets": n_bjets_M_OutsideBothJets,

            "dr_TwoFatJets": dr_two_jets, #dr_two_jets = candidatefj.delta_r(second_fj)
            "higgsMass": rec_higgs.mass,
       
            "ues_up": met.MET_UnclusteredEnergy.up.pt,
            "ues_down": met.MET_UnclusteredEnergy.down.pt,
            "pileupWeightCheck": pw_pass,

            "numberBJets_JES_down": numBJets_JES_down,
            "numberBJets_JES_up": numBJets_JES_up,
            "numberBJets_JER_down": numBJets_JER_down,
            "numberBJets_JER_up": numBJets_JER_up,
            "numberBJets_JES_FlavorQCD_up": numBJets_JES_FlavorQCD_up,
            "numberBJets_JES_FlavorQCD_down": numBJets_JES_FlavorQCD_down,
            "numberBJets_JES_RelativeBal_up": numBJets_JES_RelativeBal_up,
            "numberBJets_JES_RelativeBal_down": numBJets_JES_RelativeBal_down,
            "numberBJets_JES_HF_up": numBJets_JES_HF_up,
            "numberBJets_JES_HF_down": numBJets_JES_HF_down,
            "numberBJets_JES_BBEC1_up": numBJets_JES_BBEC1_up,
            "numberBJets_JES_BBEC1_down": numBJets_JES_BBEC1_down,
            "numberBJets_JES_EC2_up": numBJets_JES_EC2_up,
            "numberBJets_JES_EC2_down": numBJets_JES_EC2_down,
            "numberBJets_JES_Absolute_up": numBJets_JES_Absolute_up,
            "numberBJets_JES_Absolute_down": numBJets_JES_Absolute_down,
            f"numberBJets_JES_BBEC1_{self._year}_up": numBJets_JES_BBEC1_year_up,
            f"numberBJets_JES_BBEC1_{self._year}_down": numBJets_JES_BBEC1_year_down,
            f"numberBJets_JES_EC2_{self._year}_up": numBJetsJES_EC2_year_up,
            f"numberBJets_JES_EC2_{self._year}_down": numBJetsJES_EC2_year_down,

            f"numberBJets_JES_RelativeSample_{self._year}_up": numBJets_JES_RelativeSample_year_up,
            f"numberBJets_JES_RelativeSample_{self._year}_down": numBJets_JES_RelativeSample_year_down,

            f"numberBJets_JES_HF_{self._year}_up": numBJetsJES_HF_year_up,
            f"numberBJets_JES_HF_{self._year}_down": numBJetsJES_HF_year_down,
            f"numberBJets_JES_Absolute_{self._year}_up": numBJetsJES_Absolute_year_up,
            f"numberBJets_JES_Absolute_{self._year}_down": numBJetsJES_Absolute_year_down,

            "numberBJets_JES_Total_up": numBJetsJES_Total_up,
            "numberBJets_JES_Total_down": numBJetsJES_Total_down,

             "dR_genlep_recolep": dR_genlep_recolep,
             "deta": deta,
             "mjj": mjj,

             "metphi": met.phi,
             "jetvetomap": cut_jetveto,
      }

        variables = {**variables, **genVars}


        fatjetvars = {
            "fj_eta": second_fj.eta,
            "fj_phi": second_fj.phi,
            "fj_pt": second_fj.pt,
            "fj_mass": second_fj.msoftdrop,
            #"fj_mass": V_nom,
            }
        variables = {**variables, **fatjetvars}

        if self._systematics and self.isMC:
            fatjetvars_sys = {}
            for shift, vals in jec_shifted_fatjetvars["pt"].items():
                if shift != "":
                    fatjetvars_sys[f"fj_pt{shift}"] = ak.firsts(vals[VIndex])  #this is for the JEC for the V
                    #print('fj pt shift', ak.to_list(fatjetvars_sys[f"fj_pt{shift}"])[0:100]) 

#march12 - remove this and save the softdrop mass (uncorrected); will do correction after making the files
#Jan 23rd: put back in JMR/JMS
        #    for shift, vals in jmsr_shifted_fatjetvars["msoftdrop"].items():
        #        if shift != "":
        #            fatjetvars_sys[f"fj_mass{shift}"] = ak.firsts(vals)
        #            #print('fj mass shift', ak.to_list(fatjetvars_sys[f"fj_mass{shift}"])[0:100]) 

            variables = {**variables, **fatjetvars_sys}
            fatjetvars = {**fatjetvars, **fatjetvars_sys}
#****************************************************************************************************

            higgsPT_vars = { #need these for systematics, don't comment out
            "h_fj_eta": candidatefj.eta,
            "h_fj_phi": candidatefj.phi,
            "h_fj_pt": candidatefj.pt,
            "h_fj_mass": candidatefj.mass,
            }
            for shift, vals in jec_shifted_fatjetvars["pt"].items():
                if shift != "":
                    fatjetvars_sys[f"h_fj_pt{shift}"] = ak.firsts(vals[fj_idx_lep]) 
            variables = {**variables, **fatjetvars_sys}
            fatjetvars = {**fatjetvars, **fatjetvars_sys}

#MET shift
            #print('met', ak.to_list(met)[0:50])
            met_pt_sys = {}
            met_vars = ["pt"]
            for met_var in met_vars:
                for key, shift in self.jecs.items():
                #for var in ["up", "down"]:
                    for var in ["down", "up"]:
                        #print('key, value,var', key, value, var)
                        met_pt_sys[f"met_pt_{key}_{var}"] = met[shift][var][met_var]
                        #print('met[value][var].pt', met[value][var].pt)
            variables =  {**variables, **fatjetvars_sys,  **met_pt_sys}


        # Selection ***********************************************************************************************************************************************

        #only for MC! need to fix this so it applies only for MC
        self.add_selection(name = "PileupWeight", sel=pw_pass)

        for ch in self._channels:
            # trigger
            if ch == "mu":
                self.add_selection(
                    name="Trigger",
                    sel=((candidatelep.pt < 55) & trigger["mu_lowpt"]) | ((candidatelep.pt >= 55) & trigger["mu_highpt"]),
                    channel=ch,
                )
            else:
                self.add_selection(name="Trigger", sel=trigger[ch], channel=ch)

        self.add_selection(name="METFilters", sel=metfilters)
        self.add_selection(name="OneLep", sel=(n_good_muons == 1) & (n_good_electrons == 0), channel="mu")
        self.add_selection(name="OneLep", sel=(n_good_electrons == 1) & (n_good_muons == 0), channel="ele")
        self.add_selection(name="GreaterTwoFatJets", sel=(NumFatjets >= 2))

        #*************************
        fj_pt_sel = second_fj.pt > 250   
        if self.isMC:  # make an OR of all the JECs
            for k, v in self.jecs.items():
                for var in ["up", "down"]:
                    fj_pt_sel = fj_pt_sel | (second_fj[v][var].pt > 250) |  (candidatefj[v][var].pt > 250)
        self.add_selection(name="CandidateJetpT_V", sel=(fj_pt_sel == 1))
        #*************************
        #self.add_selection(name="higgs_pt", sel=(candidatefj.pt > 250))
        #self.add_selection(name="v_pt", sel=(second_fj.pt > 250))
        self.add_selection(name="LepInJet", sel=(lep_fj_dr < 0.8))
        self.add_selection(name="JetLepOverlap", sel=(lep_fj_dr > 0.03))
        self.add_selection(name="VmassCut", sel=( VCandidate_Mass > 40 )) 
        #self.add_selection(name="MET", sel=(met.pt > 30))
        #self.add_selection(name="VBF_veto", sel = (mjj < 1000 ) | (deta<3.5) )
        #self.add_selection(name="genMatching", sel = dR_genlep_recolep < 0.8)


        # gen-level matching
        signal_mask = None
        # hem-cleaning selection
        if self._year == "2018":
            hem_veto = ak.any(
                ((jets.eta > -3.2) & (jets.eta < -1.3) & (jets.phi > -1.57) & (jets.phi < -0.87)),
                -1,
            ) | ak.any(
                (
                    (electrons.pt > 30)
                    & (electrons.eta > -3.2)
                    & (electrons.eta < -1.3)
                    & (electrons.phi > -1.57)
                    & (electrons.phi < -0.87)
                ),
                -1,
            )

            hem_cleaning = (
                ((events.run >= 319077) & (not self.isMC))  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & self.isMC)
            ) & (hem_veto)

            self.add_selection(name="HEMCleaning", sel=~hem_cleaning)

# IF MC**********************************************************************************************************************************************************
        if self.isMC:
            for ch in self._channels:
                if self._year in ("2016", "2017"):
                    self.weights[ch].add(
                        "L1Prefiring",
                        events.L1PreFiringWeight.Nom,
                        events.L1PreFiringWeight.Up,
                        events.L1PreFiringWeight.Dn,
                    )
                add_pileup_weight(
                    self.weights[ch],
                    self._year,
                    self._yearmod,
                    nPU=ak.to_numpy(events.Pileup.nPU),
                )

                add_pileupid_weights(self.weights[ch], self._year, self._yearmod, jets, events.GenJet, wp="L")

                if ch == "mu":
                    add_lepton_weight(self.weights[ch], candidatelep, self._year + self._yearmod, "muon")
                elif ch == "ele":
                    add_lepton_weight(self.weights[ch], candidatelep, self._year + self._yearmod, "electron")

                ewk_corr, qcd_corr, alt_qcd_corr = add_VJets_kFactors(self.weights[ch], events.GenPart, dataset, events)
                # add corrections for plotting
                variables["weight_ewkcorr"] = ewk_corr
                variables["weight_qcdcorr"] = qcd_corr
                variables["weight_altqcdcorr"] = alt_qcd_corr

                #add top pt reweighting from farouk's repo, june 29th: 3:50 pm version
                #https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting
                if "TT" in dataset:
                    tops = events.GenPart[get_pid_mask(events.GenPart, 6, byall=False) * events.GenPart.hasFlags(["isLastCopy"])]
                    variables["top_reweighting"] = add_TopPtReweighting(tops.pt)

                if self.isSignal:
                    ew_weight = add_HiggsEW_kFactors(events.GenPart, dataset)
                    variables["EW_weight"] = ew_weight

                if self.isSignal or "TT" in dataset or "WJets" in dataset or "ST_" in dataset:
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
                    scale_weights = {}
                    if "LHEScaleWeight" in events.fields:
                        # save individual weights
                        if len(events.LHEScaleWeight[0]) == 9:
                            for i in [0, 1, 3, 5, 7, 8, 4]:
                                scale_weights[f"weight_scale{i}"] = events.LHEScaleWeight[:, i]
                    variables = {**variables, **scale_weights}

                if self.isSignal or "TT" in dataset or "WJets" in dataset or "ST_" in dataset:
                    """
                    For the PDF acceptance uncertainty:
                    - store 103 variations. 0-100 PDF values
                    - The last two values: alpha_s variations.
                    - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
                    e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
                    and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488
                    """
                    pdf_weights = {}
                    if "LHEPdfWeight" in events.fields:
                        # save individual weights
                        for i in range(len(events.LHEPdfWeight[0])):
                            pdf_weights[f"weight_pdf{i}"] = events.LHEPdfWeight[:, i]
                    variables = {**variables, **pdf_weights}

                if self.isSignal or "TT" in dataset or "WJets" in dataset or "ST_" in dataset:
                    add_ps_weight(
                        self.weights[ch],
                        events.PSWeight if "PSWeight" in events.fields else [],
                    )

                # store the final weight per ch
                variables[f"weight_{ch}"] = self.weights[ch].weight()
                if self._systematics:
                    for systematic in self.weights[ch].variations:
                        variables[f"weight_{ch}_{systematic}"] = self.weights[ch].weight(modifier=systematic)

                # store the individual weights (for DEBUG)
                for key in self.weights[ch]._weights.keys():
                    if f"weight_{key}" not in variables.keys():
                        variables[f"weight_{key}"] = self.weights[ch].partial_weight([key])

                # store b-tag weight  #i am using MEDIUM, changing to "M"
                for wp_ in ["M"]:
                    variables = {
                        **variables,
                        **get_btag_weights(
                            self._year,
                            jets,
                            bjet_selector,
                            wp=wp_,
                            algo="deepJet",
                            systematics=self._systematics,
                        ),
                    }

        # initialize pandas dataframe
        output = {}
        for ch in self._channels:
            selection_ch = self.selections[ch].all(*self.selections[ch].names)

            fill_output = True
            # for data, only fill output for the dataset needed
            if not self.isMC and self.dataset_per_ch[ch] not in dataset:
                fill_output = False

            # only fill output for that channel if the selections yield any events
            if np.sum(selection_ch) <= 0:
                fill_output = False

            if fill_output:
                out = {}
                for var, item in variables.items():
                    # pad all the variables that are not a cut with -1
                    # pad_item = item if ("cut" in var or "weight" in var) else pad_val(item, -1)
                    # fill out dictionary
                    out[var] = item

                # fill the output dictionary after selections
                output[ch] = {key: value[selection_ch] for (key, value) in out.items()}


                # fill inference
                if self._inference:
                    print('running inference')
                    for model_name in ["ak8_MD_vminclv2ParT_manual_fixwrap_all_nodes"]:
                        pnet_vars = runInferenceTriton(
                            self.tagger_resources_path,
                            events[selection_ch],
                            fj_idx_lep[selection_ch],
                            model_name=model_name,
                        )
                        pnet_df = self.ak_to_pandas(pnet_vars)
                        scores = {"fj_ParT_score": pnet_df[sigs].sum(axis=1).values}
                        #print('scores', scores)

                        hidNeurons = {}
                        for key in pnet_vars:
                            if "hidNeuron" in key:
                                hidNeurons[key] = pnet_vars[key]

                        reg_mass = {"fj_ParT_mass": pnet_vars["fj_ParT_mass"]}
                        output[ch] = {**output[ch], **scores, **reg_mass, **hidNeurons}

            else:
                output[ch] = {}

            # convert arrays to pandas
            if not isinstance(output[ch], pd.DataFrame):
                output[ch] = self.ak_to_pandas(output[ch])

            for var_ in [
                #"rec_higgs_m",
                #"rec_higgs_pt",
                "rec_V_m",
                "rec_V_pt",
            ]:
                if var_ in output[ch].keys():
                    output[ch][var_] = np.nan_to_num(output[ch][var_], nan=-1)

        # now save pandas dataframes
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")
        fname = "condor_" + fname

        for ch in self._channels:  # creating directories for each channel
            if not os.path.exists(self._output_location + ch):
                os.makedirs(self._output_location + ch)
            if not os.path.exists(self._output_location + ch + "/parquet"):
                os.makedirs(self._output_location + ch + "/parquet")
            self.save_dfs_parquet(fname, output[ch], ch)

        # return dictionary with cutflows
        return {
            dataset: {
                "mc": self.isMC,
                self._year
                + self._yearmod: {
                    "sumgenweight": sumgenweight,
                    "sumlheweight": sumlheweight,
                    "sumpdfweight": sumpdfweight,
                    "cutflows": self.cutflows,
                },
            }
        }

    def postprocess(self, accumulator):
        return accumulator
