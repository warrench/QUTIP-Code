# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:10:10 2017

@author: Chris
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


n = 2

I = qt.tensor([qt.qeye(2),qt.qeye(n),qt.qeye(n),qt.qeye(2)])
sz0 = qt.tensor([-qt.sigmaz(),qt.qeye(n),qt.qeye(n),qt.qeye(2)])
sz1 = qt.tensor([qt.qeye(2),qt.qeye(n),qt.qeye(n),-qt.sigmaz()])
a = qt.tensor([qt.qeye(2),qt.destroy(n),qt.qeye(n),qt.qeye(2)])
b = qt.tensor([qt.qeye(2),qt.qeye(n),qt.destroy(n),qt.qeye(2)])
sm1 = qt.tensor([qt.destroy(2),qt.qeye(n),qt.qeye(n),qt.qeye(2)])
sm2 = qt.tensor([qt.qeye(2),qt.qeye(n),qt.qeye(n),qt.destroy(2)])
    
Sz = 0.5*(a.dag()*a - b.dag()*b)
sigma0 = 0.5*(sz0+I)
sigma1 = 0.5*(sz1+I)

G0 = -Sz + sigma0
G1 = Sz + sigma1 - I 

def Schwinger(m,g,J):    
    
    H_m = (m/2 * 2*np.pi) * (sz0 - sz1)
    H_cav = (g/4 * 2*np.pi)*(a.dag()*a - b.dag()*b)**2
    H_int = (-J * 2 *np.pi)*(sm1.dag()*a.dag()*b*sm2 + sm1*a*b.dag()*sm2.dag())
    
    H = H_m + H_cav + H_int
    return H
    
mass = np.linspace(-30,50,321)
Eval_mat = np.zeros((len(mass),2*n*n*2))

Gauge0 = np.zeros((len(mass),2*n*n*2))
Gauge1 = np.zeros((len(mass),2*n*n*2))

LinkNumber = np.zeros((len(mass),2*n*n*2))

Evec_mat_meson = np.zeros((len(mass),2*n*n*2))
Evec_mat_string = np.zeros((len(mass),2*n*n*2))

meson = qt.tensor([qt.basis(2,1),qt.basis(n,0),qt.basis(n,1),qt.basis(2,0)])
string = qt.tensor([qt.basis(2,0),qt.basis(n,1),qt.basis(n,0),qt.basis(2,1)])

for i,m in enumerate(mass):
    if (i %((len(mass)-1)/10)) ==0:
        print ('%f Percent Completed' %(i/(len(mass)-1)*100))
    H = Schwinger(m,20,8.78)
    
    evals,evecs = H.eigenstates()
    Eval_mat[i,:] = evals
    
    Gauge0[i,:] = [qt.expect(G0**2,evecs[j]) for j in range(2*n*n*2)]
    Gauge1[i,:] = [qt.expect(G1**2,evecs[j]) for j in range(2*n*n*2)]
    
    LinkNumber[i,:] = [qt.expect(Sz,evecs[j]) for j in range(2*n*n*2)]
    
    
    for j in range(2*n*n*2):
        Number_meson = meson.dag()*evecs[j]
        Number_string = string.dag()*evecs[j]
        Evec_mat_meson[i,j] = np.abs(Number_meson[0,0])**2
        Evec_mat_string[i,j] = np.abs(Number_string[0,0])**2
        
    
    
    
np.savetxt('SchwingerEvals.txt',Eval_mat,fmt='%f',delimiter = ',')
np.savetxt('SchwingerGauge0.txt',Eval_mat,fmt='%f',delimiter = ',')
np.savetxt('SchwingerGauge1.txt',Eval_mat,fmt='%f',delimiter = ',')

for i in range(2*n*n*2):
    plt.plot(mass,-Eval_mat[:,i]/(2*np.pi))
plt.xlabel(r'Mass [MHz]')
plt.ylabel(r'Energy [MHz]')

plt.show()

for i in range(6):
    plt.plot(mass,(Eval_mat[:,i]-Eval_mat[:,0])/(2*np.pi))
plt.xlabel(r'Mass [MHz]')
plt.ylabel(r'Energy Transition [MHz]')

plt.show()

plt.plot()

#for i in range(2*n*n*2):
#    plt.plot(mass,Evec_mat_meson[:,i])
#plt.show()
#
#for i in range(2*n*n*2):
#    plt.plot(mass,Evec_mat_string[:,i])
#plt.show()



    
    
    
    
    