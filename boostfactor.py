#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial  import KDTree
import mycosmology as cosmos

#-----Cosmology parameters(PLANCK18)--------
pi      = np.pi
H0      = 67.4
h       = 0.674
w       = -1.0
omega_m = 0.315
omega_l = 0.685
sigma8  = 0.811
omega_k = 1.0-omega_m-omega_l
rho_crt0= 2.78e11                # M_sun Mpc^-3 *h*h 
rho_bar0= rho_crt0*omega_m       # M_sun Mpc^-3 *h*h
ns      = 0.95
alphas  = -0.04

c     = 2.9970e5                # Speed of light in unit of km/s
G     =6.67408e-11              # Gravitational constant in unit of N*m^2/kg
ckm   =3.240779e-17             # 1 km equals 3.240779e-17 kpc/h
ckg   =5.027854e-31             # 1 kg equals 5.027854e-31 solar mass
dH0   =c/H0                     # Hubble distance in Mpc/h
fac   =(c*c*ckg)/(4.*pi*G*ckm)  # The value of c^2/(4.*pi*G)


#----------------------------------------------
def angulardis2(z1,z2):
   dis   = np.zeros(len(z2))
   for i in range(len(z2)):
      dls  = cosmos.Da2(z1,z2[i])
      dis[i]  = dls
   return dis

def boostfactor(posl,poss):
#---projected radial bins---------------------------
   Nbins  = 10
   rlow   = 0.02
   rhig   = 2.0
   rbounds= np.zeros(Nbins+1)
   tmp    = (np.log10(rhig)-np.log10(rlow))/Nbins
   for ir in range(Nbins+1):
      ytmp= np.log10(0.04)+float(ir)*tmp
      rbounds[ir]=10.0**ytmp
#---positions and kdtree structure------------------
   ra,dec,zl,dl      = posl
   nlens             = len(ra) 
   ras,decs,zs,ds,wt = poss
   pos  = np.column_stack([ras,decs])
   trees= KDTree(pos)
#---measure boostfactor----------------------------
   ngals= np.zeros((Nbins))
   for i in range(nlens):
       idx = trees.query_ball_point([ra[i],dec[i]],0.5)
       dls = angulardis2(zl[i],zs[idx])
       Sigc= fac*ds[idx]/(dl[i]*dls)/(1.0+zl[i])/(1.0+zl[i])
       xm0 = np.cos(pi/2.0-decs[idx]*pi/180.0)
       xm1 = np.cos(pi/2.0-dec[i]*pi/180.0)
       xm2 = np.sin(pi/2.0-decs[idx]*pi/180.0)
       xm3 = np.sin(pi/2.0-dec[i]*pi/180.0)
       xm4 = np.cos((ras[idx]-ra[i])*pi/180.0)
       the = np.arccos(xm0*xm1+xm2*xm3*xm4)
       tps = (((ras[idx]-ra[i])*np.cos(dec[i]*pi/180.0))*pi/180.0)
       tpc = (decs[idx]-dec[i])*pi/180.0
       cph = (2.0*tps*tps)/the/the-1.0
       sph = (2.0*tps*tpc)/the/the
       Rp  = dl[i]*the*(1.0+zl[i])
       wtt = wt[idx]*Sigc
       for j in range(Nbins):
          ixa = Rp>=rbounds[j]
          ixb = Rp<=rbounds[j+1]
	  ixx = ixa&ixb
	  ngals[j]=ngals[j]+wtt[ixx].sum()*len(dls[ixx])
   
   return ngals 

def main():
  import sys
  flens  = str(sys.argv[1])
  fsrcs  = str(sys.argv[2])
  ra,dec,zl,dl = np.loadtxt(flens,unpack=True,skiprows=1)
  ras,decs,zs,ds,e1,e2,wt = np.loadtxt(fsrcs,unpack=True,skiprows=1)
  posl   = [ra,dec,zl,dl]
  poss   = [ras,decs,zs,ds,wt]
  bfc    = boostfactor(posl,poss)
  print bfc
if __name__=='__main__':
   main()

