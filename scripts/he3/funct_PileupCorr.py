#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:47:48 2020

@author: francescopiscitelli
"""

import numpy as np

def eventRatePileUpCorrection(measuredRate, shapingTime):
    
    # measuredRate = [1e3, 1e4, 1e5]
    # shapingTime = 1e-6

    # input Hz and s, measuredRate a list and shapingTime a float
    
    pulseFWHM = 2.355*shapingTime
    
    expi = np.zeros(len(measuredRate))
    for k in range(len(measuredRate)):
        expi[k] = np.exp(-pulseFWHM*measuredRate[k])
    
    pileupNumOfEv = np.array([0,1,2,3,4])
    
    ratioPileupEv = np.zeros((len(measuredRate),len(pileupNumOfEv)))
    
    for k in range(len(measuredRate)):
        for j in range(len(pileupNumOfEv)):
            ratioPileupEv[k,j] = expi[k]*((1-expi[k])**pileupNumOfEv[j])
    
    correctionFactor = np.zeros(len(measuredRate))
    realRate         = np.zeros(len(measuredRate))
    
    for k in range(len(measuredRate)):
        correctionFactor[k] = np.sum( (pileupNumOfEv+1)*ratioPileupEv[k,:] )
        realRate[k]         = correctionFactor[k]*measuredRate[k]

    return realRate, correctionFactor, ratioPileupEv
