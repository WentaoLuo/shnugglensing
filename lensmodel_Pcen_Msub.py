#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy import interpolate

###############PART I ###############################################
# Basic Parameters---------------------------
h       = 0.672
w       = -1.0
omega_m = 0.315
omega_l = 0.685
omega_k = 1.0-omega_m-omega_l
rho_crt0= 2.78e11                # M_sun Mpc^-3 *h*h 
rho_bar0= rho_crt0*omega_m       # M_sun Mpc^-3 *h*h
pi      = np.pi
ns      = 0.965
alphas  = -0.04
sigma8  = 0.811
rrp,esdth= np.loadtxt('twohaloesd.dat',unpack=True)

# Read the tables for interpolation---------- 
fname,Rc1,Rc2 = np.loadtxt('r_c-range.tsv',\
                dtype=np.str,usecols=(0,1,2),\
                unpack=True)
nx  = len(fname) 
ny  = 3077 
rc1 = np.zeros(nx)
rc2 = np.zeros(nx)
tabs= np.zeros((nx,ny,2))
for i in range(nx):
  rc1[i] = float(Rc1[i])
  rc2[i] = float(Rc2[i])
  fsub   = 'mean/'+fname[i]
  Rps,dsig = np.loadtxt(fsub,dtype=np.str,usecols=(0,1),unpack=True)
  for j in range(len(Rps)):
     tabs[i,j,0] = Rps[j] 
     tabs[i,j,1] = dsig[j]

# Generate Projected distance-------------
Rmax = 3.0
Rmin = 0.01
Nbin = 5 
rbin = np.zeros(Nbin+1)
r    = np.zeros(Nbin)
xtmp = (np.log10(Rmax)-np.log10(Rmin))/Nbin
for i in range(Nbin):
  ytmp1 = np.log10(0.01)+float(i)*xtmp
  ytmp2 = np.log10(0.01)+float(i+1)*xtmp
  #ytmp1 = np.log10(0.05)+float(i)*xtmp
  #ytmp2 = np.log10(0.05)+float(i+1)*xtmp
  rbin[i] = 10.0**ytmp1
  rbin[i+1] = 10.0**ytmp2
  r[i] =(rbin[i])*1./2.+(rbin[i+1])*1.0/2.0
  #r[i] =(rbin[i])*1./3.+(rbin[i+1])*2.0/3.0
# Get the closest Rp from tabs-------------
tmx  = np.linspace(1,ny,ny)
tmx  = tmx.astype(np.int)
rtmp = tabs[0,:,0]
#-------------------------------------------
#cluster redshift
zl  = 1.2
#M0  = 0.0060687
#r   = r[0:8]
#print r
twoesd= np.interp(r,rrp,esdth)
############## PART II ######################################
def galaxybias(logM):
  #fac = (1.0+1.17*corr)**1.49/(1.0+0.69*corr)**2.09
  Mnl = 8.73*10e+11
  Mh  = 10.0**logM
  xx  = Mh/Mnl
  b0  = 0.53+0.39*xx**0.45+(0.13/(40.0*xx+1.0))\
          +5.0*0.0004*xx**1.5
  bias= b0+\
        np.log10(xx)*(0.4*(omega_m-0.3+ns-1)+\
        0.3*(sigma8-0.9+h-0.7)+0.8*alphas)
  return 0.0+bias

#-----------------------------------------------
def nfwfuncs(rsub):
     x   = r/rsub
     x1  = x*x-1.0
     x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
     x3  = np.sqrt(np.abs(1.0-x*x))
     x4  = np.log((1.0+x3)/(x))
     s1  = r*0.0
     s2  = r*0.0
     ixa = x>0.
     ixb = x<1.0
     ix1 = ixa&ixb
     s1[ix1] = 1.0/x1[ix1]*(1.0-x2[ix1]*x4[ix1])
     s2[ix1] = 2.0/(x1[ix1]+1.0)*(np.log(0.5*x[ix1])\
               +x2[ix1]*x4[ix1])

     ix2 = x==1.0
     s1[ix2] = 1.0/3.0
     s2[ix2] = 2.0+2.0*np.log(0.5)

     ix3 = x>1.0
     s1[ix3] = 1.0/x1[ix3]*(1.0-x2[ix3]*np.arctan(x3[ix3]))
     s2[ix3] = 2.0/(x1[ix3]+1.0)*(np.log(0.5*x[ix3])+\
              x2[ix3]*np.arctan(x3[ix3]))

     res = s2-s1
     return res
#-----------------------------------------------
def haloparams(logM,con):
   ccon      = con*(10.0**(logM-14.0))**(-0.11)
   efunc     = 1.0/np.sqrt(omega_m*(1.0+zl)**3+\
              omega_l*(1.0+zl)**(3*(1.0+w))+\
              omega_k*(1.0+zl)**2)
   rhoc      = rho_crt0/efunc/efunc
   omegmz    = omega_m*(1.0+zl)**3*efunc**2
   ov        = 1.0/omegmz-1.0
   dv        = 18.8*pi*pi*(1.0+0.4093*ov**0.9052)
   rhom      = rhoc*omegmz

   r200 = (10.0**logM*3.0/200./rhom/pi)**(1./3.)
   rs   = r200/ccon
   delta= (200./3.0)*(con**3)\
               /(np.log(1.0+con)-con/(1.0+con))

   amp  = 2.0*rs*delta*rhoc*10e-14
   res  = np.array([amp,rs,r200])

   return res
#----------------------------------------------
def NFWcen(theta):
  logM,con,Rsig = theta
  amp,rs,r200   = haloparams(logM,con)
  res           = amp*nfwfuncs(rs) 
  return res
#-------------------------------------------------
def ESDRsig(theta,Rp):
  logM,con,Rsig,frac = theta
  amp,rs,r200   = haloparams(logM,con) 
  itx           = r<=r200
  twoesd[itx]   = 0.0
  ratio         = Rsig/rs
  xx            = np.linspace(1,48,48)-1
  idx           = np.abs(rc2-ratio)==np.min(np.abs(rc2-ratio))  
  inx           = int(xx[idx])
  summ          = np.zeros(3077)
  res           = np.zeros(len(Rp))
  sat           = np.zeros(len(Rp))
  nfwesd        = amp*nfwfuncs(rs)
  for i in range(inx):
     tmp  = amp*tabs[i,:,1]*(rc2[i]-rc1[i])/rc2[inx]
     summ = summ+tmp

  for j in range(len(Rp)):
     idx    = np.abs(rtmp*rs-r[j])==np.min(np.abs(rtmp*rs-r[j]))
     ind    = int(tmx[idx][0])
     #intx   = np.array([ind-2,ind-1,ind,ind+1,ind+2])
     intx   = np.array([ind-1,ind,ind+1])
     rrtmp  = (rtmp[intx]*rs)
     totr   = np.sum(rrtmp)
     sstmp  = summ[intx]
     ff     = interpolate.interp1d(rrtmp,sstmp,kind='quadratic')
     rinterp= np.array([rrtmp[0]+0.0001,r[j],rrtmp[1]+0.0001,rrtmp[2]+0.0001,rrtmp[3]-0.0001])
     #rinterp= np.array([rrtmp[0]+0.0001,r[j],rrtmp[1]+0.0001])
     restmp = ff(rinterp)
     sat[j] = (frac)*restmp[1]
     res[j] = (1.0-frac)*nfwesd[j]+(frac)*restmp[1]
  #struct = {'amp':amp,'res':res,'Cen':(1.0-frac)*nfwesd,'Sat':(frac)*restmp}
  struct = {'amp':amp,'res':res,'Cen':(1.0-frac)*nfwesd,'Sat':sat,'Twohalo':twoesd}
 
  return struct
#----------------------------------------------
def lnprior(theta):
  logM,con,Rsig,M0,frac = theta
  if 12.0<logM<15.0 and 1.0<con<10.0 and 0.00<Rsig<0.8\
         and 0.0<M0<1.0 and 0.0<frac<1.0:
         #and 0.0<frac<1.0:
     return 0.0
  return -np.inf
#---------------------------------------------
def lnlike(theta,Rp,esd,err):
  logM,con,Rsig,M0,frac = theta
  #logM,con,Rsig,frac = theta
  theta1   = np.array([logM,con,Rsig,frac])
  bias     = galaxybias(logM)
  #stellar  = M0/2.0/pi/r/r
  #stellar  = M0/2.0/pi/Rp/Rp
  stellar  = M0/pi/Rp/Rp
  nrr      = len(Rp)
  #stars    = stellar[0:nrr]
  stars    = stellar
  struct= ESDRsig(theta1,Rp)
  model = stars+struct["res"]+bias*struct["Twohalo"]
  invers= 1.0/err/err
  #print len(model),len(esd),len(err),len(Rp)
  diff  = -0.5*((esd-model)**2*invers-np.log(invers))
  return diff.sum()

#-----------------------------------------------------
def lnprob(theta,Rp,esd,err):
  lp = lnprior(theta)
  #print len(r),len(Rp),len(esd),len(err)
  if not np.isfinite(lp):
     return -np.inf
  return lp+lnlike(theta,Rp,esd,err)
#----------------------------------------------

   
############ PART III ############################################
def main():
  import sys
#-----------Read the data-----------------------------------
  print sys.argv[1]
  if int(sys.argv[1]) == 0:
	  fname = 'esd_erosita/DelSig_xray_hsc_3.dat'
	  dat = np.loadtxt(fname,unpack=True) 
	  
	  rp  = dat[7,:]
	  esd = dat[5,:]
	  err = dat[11,:]
	  
	  logM= 13.0
	  con = 8.0
	  Rsig= 0.2
	  M0  = 0.006
	  Pcen= 0.5
#---------------------------------------------------------
	  pars= np.array([logM,con,Rsig,M0,Pcen])
	  #print rp
	  ndim,nwalkers = 5,200
	  pos = [pars+1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	  sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(rp,esd,err))
	  sampler.run_mcmc(pos,4000)
	  
	  burnin = 1000
	  samples=sampler.chain[:,burnin:,:].reshape((-1,ndim))  
	  Mh,cn,Roff,Mst,Pc = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples,[16,50,84],axis=0)))
	  #Mh,cn,Roff,Pc = map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples,[16,50,84],axis=0)))
	  print 'logM: ',Mh
	  print 'c: ',cn
	  print 'Rsig: ',Roff
	  print 'M0: ',Mst
	  print 'Psat: ',Pc
	  #fig = corner.corner(samples,labels=["logM","c","Rsig","fsat"],\
	#	    truths=[Mh[0],cn[0],Roff[0],Pc[0]],\
	#	    plot_datapoints=False,plot_density=True,color='b')
	  fig = corner.corner(samples,labels=["logM","c","Rsig","M0","fsat"],\
		    truths=[Mh[0],cn[0],Roff[0],Mst[0],Pc[0]],\
		    plot_datapoints=False,plot_density=True,color='b')
	  plt.savefig('mcmc_gal_ii.eps')
	  plt.show()
#------------plot the model vs data----------------------------------------
  if int(sys.argv[1]) == 1:
	  fname = 'esd_erosita/DelSig_xray_hsc_3.dat'
	  dat = np.loadtxt(fname,unpack=True) 
	  rp    = dat[7,:]
	  esda  = dat[5,:]
	  erra  = dat[11,:]
	  plt.errorbar(rp,esda,yerr=erra,fmt='ko',ms=10.0,elinewidth=2.5,label='Control all')
#------------------------------------------------------------
	  paramsa   = np.array([13.26,6.7,0.65,0.69])
	  starsa    = 0.0999/pi/r/r
	  structa   = ESDRsig(paramsa,r)
	  #bias      = galaxybias(13.26)
	  #modela    = starsa+structa["res"]+bias*structa['Twohalo']
	  nfwa      = structa["Cen"]
	  #sata      = structa["Sat"]
	  #twoa      = bias*structa['Twohalo']
#-------------------------------------------------------------
	  plt.xscale('log')
	  plt.yscale('log')
	  plt.xlim([0.015,2.])
	  plt.ylim([0.01,500.])
	  plt.plot(r,modela,'k-',linewidth=3.0,label='All')
	  plt.plot(r,starsa,'k:',linewidth=3.0,label='Stellar')
	  plt.plot(r,nfwa,'k-.',linewidth=3.0,label='Central')
	  plt.plot(r,sata,'k--',linewidth=3.0,label='Satellite')
	  plt.plot(r,twoa,'b:',linewidth=3.0,label='Two halo')
	  plt.legend()
	  plt.show()
#-------------------------------------------------------------
	  fname = 'esd_gal_i/DelSig_gal_i_s.dat'
	  dat = np.loadtxt(fname,unpack=True) 
	  esdi  = dat[5,:]
	  erri  = dat[11,:]
	  fname = 'esd_gal_ii/DelSig_gal_ii_s.dat'
	  #fname = 'esd_type_ii/DelSig_bin1.dat'
	  dat = np.loadtxt(fname,unpack=True) 
	  esdii = dat[5,:]
	  errii = dat[11,:]

	  paramsa   = np.array([13.26,6.7,0.65,0.69])
	  starsa    = 0.0999/pi/r/r
	  structa   = ESDRsig(paramsa,r)
	  bias      = galaxybias(13.26)
	  modela    = starsa+structa["res"]+bias*structa['Twohalo']
	  nfwa      = structa["Cen"]
	  sata      = structa["Sat"]
	  twoa      = bias*structa['Twohalo']
	  print rp, sata
	  paramsi   = np.array([13.293,6.48,0.66,0.62])
	  starsi    = 0.106/pi/r/r
	  structi   = ESDRsig(paramsi,r)
	  bias      = galaxybias(13.292)
	  modeli    = starsa+structa["res"]+bias*structi['Twohalo']
	  #modeli    = starsi+structi["res"]
	  nfwi      = structi["Cen"]
	  sati      = structi["Sat"]
	  twoi      = bias*structi['Twohalo']

	  paramsii  = np.array([13.148,5.84,0.593,0.63])
	  starsii   = 0.095/pi/r/r
	  structii  = ESDRsig(paramsii,r)
	  bias      = galaxybias(13.148)
	  modelii   = starsa+structa["res"]+bias*structii['Twohalo']
	  #modelii   = starsii+structii["res"]
	  nfwii     = structii["Cen"]
	  satii     = structii["Sat"]
	  twoii     = bias*structii['Twohalo']

	  fig,axs=plt.subplots(nrows=1,ncols=3,sharex=True,
			  sharey=False,figsize=(15,5))

	  ax1=axs[0]
	  ax1.set_xscale('log')
	  ax1.set_yscale('log')
	  ax1.set_xlim([0.015,2.])
	  ax1.set_ylim([0.5,900.])
	  ax1.errorbar(rp,esda,yerr=erra,fmt='ko',ms=10.0,elinewidth=2.5,label='Control all')
	  ax1.plot(r,modela,'k-',linewidth=3.0,label='All')
	  ax1.plot(r,starsa,'k:',linewidth=3.0,label='Stellar')
	  ax1.plot(r,nfwa,'k-.',linewidth=3.0,label='Central')
	  ax1.plot(r,sata,'k--',linewidth=3.0,label='Satellite')
	  ax1.plot(r,twoa,'b:',linewidth=3.0,label='Two halo')
	  ax1.legend()

	  ax2=axs[1]
	  ax2.set_xscale('log')
	  ax2.set_yscale('log')
	  ax2.set_xlim([0.015,2.])
	  ax2.set_ylim([0.5,900.])
	  ax2.errorbar(rp,esdi,yerr=erri,fmt='ko',ms=10.0,elinewidth=2.5,label='Control I')
	  ax2.plot(r,modeli,'k-',linewidth=3.0)
	  ax2.plot(r,starsi,'k:',linewidth=3.0)
	  ax2.plot(r,nfwi,'k--',linewidth=3.0)
	  ax2.plot(r,sati,'k--',linewidth=3.0)
	  ax2.plot(r,twoi,'b:',linewidth=3.0)
	  ax2.set_yticks(())
	  ax2.legend()

	  ax3=axs[2]
	  ax3.set_xscale('log')
	  ax3.set_yscale('log')
	  ax3.set_xlim([0.015,2.])
	  ax3.set_ylim([0.5,900.])
	  ax3.errorbar(rp,esdii,yerr=errii,fmt='ko',ms=10.0,elinewidth=2.5,label='Control II')
	  ax3.plot(r,modelii,'k-',linewidth=3.0)
	  ax3.plot(r,starsii,'k:',linewidth=3.0)
	  ax3.plot(r,nfwii,'k--',linewidth=3.0)
	  ax3.plot(r,satii,'k--',linewidth=3.0)
	  ax3.plot(r,twoii,'b:',linewidth=3.0)
	  ax3.set_yticks(())
	  ax3.legend()
	  #ax3.spines['right'].set_linewidth(3)


	  fig.text(0.5,0.01,r'$\mathbf{R h^{-1}Mpc}$',ha='center',size=16)
	  fig.text(0.05,0.5,r'$\mathbf{ESD (M_{\odot}/pc^2)}$',va='center',size=16,rotation='vertical')

	  plt.subplots_adjust(wspace = 0.0, hspace = 0.0 )
	  fig.savefig('fgal.eps')
	  plt.show()
	  

#------------------------------------------------------------

if __name__=='__main__':
   main()
