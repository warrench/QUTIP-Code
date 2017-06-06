# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:20:49 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

def JC(Ej,Ec,wr,phi,n=2):
    N = 10
    a = qt.tensor(qt.destroy(N),qt.qeye(n))
    sm = qt.tensor(qt.qeye(N),qt.destroy(n))
    
    EJ = Ej*np.abs(np.cos(phi*2*np.pi))*np.sqrt(1.0+(0.025*(np.tan(phi*2*np.pi)))**2.0)
    w0 = np.sqrt(8*EJ*Ec)
    wr = wr * (2*np.pi)
    
    g = 0.1 * (2*np.pi)
    if n==2:
        w01 = (w0-Ec)*(2*np.pi)
        H0 = w01*sm.dag()*sm + wr*a.dag()*a
        Hint = g*(a.dag()*sm + a*sm.dag())
        H = H0 + Hint
    elif n>2:
       H0 = (w0*(2*np.pi))*sm.dag()*sm - (Ec/12*(2*np.pi))*(sm+sm.dag())**4 + wr*a.dag()*a
       Hint = g*(a.dag()*sm + a*sm.dag())
       H = H0 + Hint
    else:
        H = np.nan
       
    return H, Hint/g

phi = np.linspace(-0.5,0.5,10000)

Eval_mat1 = np.zeros((len(phi),2*10))
Eval_mat2 = np.zeros((len(phi),2*10))

expect1 = []
expect2 = []


   
for i,Phi in enumerate(phi):
    H1,Hint = JC(16,0.37,6,Phi)
    #H2 = JC(16,0.37,7,Phi)
    H1.tidyup(atol=1e-1)
    
    evals1,evecs1 = H1.eigenstates()
    gnd = evecs1[0]
    
    first = evecs1[1]
    second = evecs1[2]
    
    #gnd_to_first = gnd*first.dag()
    #gnd_to_first = 0.5*(gnd_to_first + gnd_to_first.dag())
    
    #first_to_second = first*second.dag()
    #first_to_second = 0.5*(first_to_second + first_to_second.dag())
    expect1.append(qt.expect(Hint,first*first.dag()))
    expect2.append(qt.expect(Hint,second*second.dag()))
    #expect1.append(np.abs(qt.expect(Hint,gnd_to_first)))
    #expect2.append(np.abs(qt.expect(Hint,first_to_second)))
    
    
    #evals2 = H2.eigenenergies()
    
    Eval_mat1[i,:] = np.real(evals1)
    #Eval_mat2[i,:] = np.real(evals2)
print(first)
#print(min(expect2))
for i in range(3):
    plt.plot(phi,(Eval_mat1[:,i]-Eval_mat1[:,0])/(2*np.pi))
plt.xlabel(r'$\frac{\Phi}{2\pi}$')
plt.ylabel('Frequency [GHz]')
plt.title('$\omega_{01} > \omega_{r}$')
plt.show()
#for i in range(3):
#    plt.plot(phi,(Eval_mat2[:,i]-Eval_mat2[:,0])/(2*np.pi))
#plt.xlabel(r'$\frac{\Phi}{2\pi}$')
#plt.ylabel('Frequency [GHz]')
#plt.title('$\omega_{01} < \omega_{r}$')
#plt.show()

plt.plot(phi,expect1,phi,expect2)
plt.show()