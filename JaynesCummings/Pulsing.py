# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:25:11 2017

@author: Chris
"""

import qutip as qt
import matplotlib.pyplot as plt
from numpy import sqrt, linspace, pi,exp


wc = 1.0  * 2 * pi  # cavity frequency
wa = 1.0  * 2 * pi  # atom frequency
g  = 0.05 * 2 * pi  # coupling strength
kappa = 0.005       # cavity dissipation rate
gamma = 0.05        # atom dissipation rate
N = 15              # number of cavity fock states
n_th_a = 0.0        # avg number of thermal bath excitation
use_rwa = True

tlist = linspace(0,100,10000)

qt.Options(nsteps=1e10)

# intial state
psi0 = qt.tensor(qt.basis(N,0), qt.basis(2,0))    # start with an excited atom

e_cav = qt.tensor(qt.fock_dm(N,1),qt.fock_dm(2,0))
e_qb = qt.tensor(qt.fock_dm(N,0),qt.fock_dm(2,1))

# operators
a  = qt.tensor(qt.destroy(N), qt.qeye(2))
sm = qt.tensor(qt.qeye(N), qt.destroy(2))

# Hamiltonian
if use_rwa:
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
else:
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())

c_ops = []

# cavity relaxation
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_ops.append(sqrt(rate) * a)

# cavity excitation, if temperature > 0
rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(sqrt(rate) * a.dag())

# qubit relaxation
rate = gamma
if rate > 0.0:
    c_ops.append(sqrt(rate) * sm) 

Drive_amp = 10
std_G = 5

pulse_shape_G = Drive_amp*exp(-(tlist - 5)**2/(2*std_G**2))

def pulse_shape(t,args):
    return Drive_amp*exp(-(t - 5)**2/(2*std_G**2))
    
H_pulse = a + a.dag()
H_tot = [H, [H_pulse,pulse_shape]]

output = qt.mesolve(H_tot,psi0,tlist,c_ops,[e_cav,e_qb],progress_bar=True)
#output = qt.mesolve(H_tot,psi0,tlist,c_ops,[a.dag()*a,sm.dag()*sm],progress_bar=True)
plt.plot(output.times,output.expect[0])
plt.plot(output.times,output.expect[1])
plt.plot(tlist, pulse_shape_G/Drive_amp)

plt.xlabel(r'Time')
plt.ylabel(r'Occupation Probability')
plt.legend([r'|1g>',r'|0e>',r'Pulse Shape [Normalized Amplitude]'])
plt.title(r'Initial State |0g>')
plt.show()

    
    