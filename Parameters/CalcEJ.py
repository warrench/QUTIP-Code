# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:40:18 2017

@author: Chris
"""

import numpy as np

def CalcEj(Rn):
    h = 6.626e-34
    echarge = 1.609e-19
    kb = 1.38065e-23
    
    delta = 1.76*kb*1.176  #1.76*kb*Tc
    #Rq = h/(4*echarge**2)  # R_q = h/4e^2
    Rq_scaled = 1/(4*echarge**2)
    
    Ej = (Rq_scaled*delta)/(2*Rn)/1e9
    
    return Ej

print(CalcEj(10e3))
print(CalcEj(16e3))