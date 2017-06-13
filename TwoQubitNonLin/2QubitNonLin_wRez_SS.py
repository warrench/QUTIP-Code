# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:55:32 2017

@author: Chris
"""

import numpy as np
#import matplotlib.pyplot as plt
import qutip as qt


numcores = 8

opts=qt.Options()
opts.num_cpus = numcores

#==============================================================================
#                         Define Two Qubit Hamiltonian
#==============================================================================

n = 2 #number of oscillator states
def TwoQubitNonLin_NoInt(EjL,EjR,CL,CR,Ej,Cj,phi=0):
               
    
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))
    
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    
    a = qt.tensor(qt.destroy(n),qt.qeye(n))
    b = qt.tensor(qt.qeye(n),qt.destroy(n))
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H0 = H_L + H_R
    H = H_L + H_R + H_LR
    
    return H, H0, H_LR

def TwoQubitNonLin_wInt(EjL,EjR,CL,CR,Ej,Cj,phi=0):
    
    e_charge = 1.602e-19 #electron charge
    h = 6.626e-34        #planck's constant
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))
    
    C_to_GHz = (e_charge**2)/(h*1e9)   # e^2/(hC)  gives frequency in GHz
    
    C_det = 1.0/(Cj*CL + Cj*CR + CL*CR)
    eps_L = 2*np.sqrt((Cj+CR)*C_det*C_to_GHz/(EJ+EjL))
    eps_R = 2*np.sqrt((Cj+CL)*C_det*C_to_GHz/(EJ+EjR))
    
    a = qt.tensor([qt.destroy(n),qt.qeye(n),qt.qeye(5)])
    b = qt.tensor([qt.qeye(n),qt.destroy(n),qt.qeye(5)])
    c = qt.tensor([qt.qeye(n),qt.qeye(n),qt.destroy(5)])
    
    
    wr = 6.0 * (2*np.pi)
    lambda1 = 0.1 * (2*np.pi)    
    lambda2 = 0.15 * (2*np.pi)
    
    H_lambda = lambda1*(a.dag()*c + a*c.dag()) + lambda2*(b.dag()*c + b*c.dag())
    
    #phi_L = np.sqrt(eps_L/2.0)*(a+a.dag())
    #phi_R = np.sqrt(eps_R/2.0)*(b+b.dag())
    
    H_L = ((EJ+EjL)*eps_L * (2*np.pi))*a.dag()*a
    H_R = ((EJ+EjR)*eps_R * (2*np.pi))*b.dag()*b
    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b + a*b.dag())
            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a*b.dag()))
#    H_LR = (((2*Cj*C_det*C_to_GHz/np.sqrt(eps_L*eps_R)) * (2*np.pi))*(a.dag()*b - a.dag()*b.dag() + a*b.dag() - a*b)
#            - (EJ*np.sqrt(eps_L*eps_R)/2 * (2*np.pi))*(a.dag()*b + a.dag()*b.dag() + a*b.dag() + a*b))
    
    H0 = H_L + H_R + H_LR + wr*c.dag()*c
    
    H_int1 = -(2*np.pi)*(np.exp(-eps_L/4)*EjL*eps_L**2/16.0)*((a.dag()**2)*(a**2))
    H_int2 = -(2*np.pi)*(np.exp(-eps_R/4)*EjR*eps_R**2/16.0)*((b.dag()**2)*(b**2))
    H_int3 = -(2*np.pi)*(np.exp(-np.sqrt(eps_L*eps_R)/4)/16.0)*((eps_L**2)*(a.dag()**2)*(a**2) + (eps_R**2)*(b.dag()**2)*(b**2) + 4*eps_L*eps_R*(a.dag()*a)*(b.dag()*b) )
    H_int = H_int1 + H_int2 + H_int3
    
    #H_int = -(2*np.pi*EJ)*(phi_R-phi_L).cosm() - (EJ*np.pi)*(phi_R-phi_L)**2 - (2*np.pi*EjR)*phi_R.cosm() - (EjR*np.pi)*(phi_R**2) - (EjL*2*np.pi)*phi_L.cosm() - (EjL*np.pi)*(phi_L**2)
    H = H0 + H_int + H_lambda
    
    return H


#==============================================================================
#------------------------------------------------------------------------------
#==============================================================================
#                              Steady State
#==============================================================================

#H = TwoQubitNonLin_wInt(17.0,16.0,65e-15,48.8e-15,20.0,40.0e-15,-0.15)
a = qt.tensor([qt.destroy(n),qt.qeye(n),qt.qeye(5)])
b = qt.tensor([qt.qeye(n),qt.destroy(n),qt.qeye(5)])
c = qt.tensor([qt.qeye(n),qt.qeye(n),qt.destroy(5)])

kappa1 = 0.04
kappa2 = 0.05
kappa3 = 40e-6
    
c_ops = [np.sqrt(kappa1)*a,np.sqrt(kappa2)*b,np.sqrt(kappa3)*c]

def H_t(t,args):
    return args['H0'] + args['Hdr_p']*np.exp(-1j*args['wd']*t) + args['Hdr_m']*np.exp(1j*args['wd']*t)

def task(w, H, c):
    T = 200
    Hdr_p = (0.0001*2*np.pi)*c.dag()
    Hdr_m = (0.0001*2*np.pi)*c
    Hargs = {'H0': H, 'wd': w, 'Hdr_p': Hdr_p, 'Hdr_m': Hdr_m}
    U = qt.propagator(H_t,T,c_ops,Hargs)
    rho_ss = qt.propagator_steadystate(U)
    I = (1/np.sqrt(2))*(c + c.dag())
    Q = (1j/np.sqrt(2))*(c - c.dag())
    Iavg = qt.expect(I,rho_ss)
    Qavg = qt.expect(Q,rho_ss)
    Mag = np.sqrt(Iavg**2 + Qavg**2)
    try:
        Phase = np.arctan2(Iavg,Qavg)*180/np.pi
    except:
        Phase = np.nan
    Pow = qt.expect(I**2+Q**2,rho_ss)
    
    return [Pow,Mag,Phase]

def run():
    N=101
    M=101
    
    A_pow = np.zeros((M,N))
    A_mag = np.zeros((M,N))
    A_phase = np.zeros((M,N))
    
    
    phi = np.linspace(-0.20,-0.10,N)
    wd = (2*np.pi) * np.linspace(5.5,6.5,M)
    
    
    for j in range(N):
        print(j)
        H = TwoQubitNonLin_wInt(17.0,16.0,65e-15,48.8e-15,20.0,40.0e-15,phi[j])
        
        results = qt.parallel_map(task,wd,task_args=(H,c),progress_bar=True)
        results = np.array(results)
        A_pow[j,:] = results[:,0]
        A_mag[j,:] = results[:,1]
        A_phase[j,:] = results[:,2]
        
    np.savetxt('Mag2D.txt',A_mag,fmt='%f',delimiter = ',')
    np.savetxt('Phase2D.txt',A_phase,fmt='%f',delimiter = ',')
    np.savetxt('Power2D.txt',A_pow,fmt='%f',delimiter = ',')
    
if __name__=='__main__':
    run()
        
    