pro boost_1d_galaxy

;This is a program to calculate the shear from galaxy sample.
;Parameters;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;-----------------------------------------------------------
c     =300000.0                ; Speed of light in unit of km/s
pi    =3.14159265              ; Pi as circumference ratio
G     =6.754e-11               ; Gravitational constant in unit of N*m^2/kg
ckm   =3.240779e-17            ; 1 km equals 3.240779e-17 kpc/h
ckg   =5.027854e-31            ; 1 kg equals 5.027854e-31 solar mass
H0    =67.3                    ; Hubble constant at z=0 frim PLANK space mission in unit of km/s/Mpc
dH0   =c/H0                    ; Hubble distance in Mpc/h
fac   =(c*c*ckg)/(4.*pi*G*ckm) ; The value of c^2/(4.*pi*G)

;print,dH0,fac
;------------------------------------------------------------
Nsamp =200                    ; Number of bootstrap samples
Rmax  =2.0                    ; Maximum of the radius to measure
Rmin  =0.02                  ; Minimum of the radius to measure 
Nbin  =10                     ; N bins of the signals
rbin  =fltarr(11)             ; Radius of each bin

xtmp  =(alog10(Rmax)-alog10(Rmin))/Nbin
for ir=0,Nbin do begin
	ytmp    =alog10(0.02)+float(ir)*xtmp
	rbin(ir)=10.0^ytmp
;	print,rbin(ir)
endfor

;Start to read files;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;-------------------------------------------------------------
print,'% Start to load lens file... ...'
;fname =strcompress('Lb'+string(2),/remove_all)
;fname ='lwtlens.dat'
fname ='lenran.dat'
nl    =file_lines(fname)
data  =dblarr(4,nl) 
openr,lunr,fname,/get_lun
readf,lunr,data
free_lun,lunr

ral   =data(0,*) 
decl  =data(1,*) 
zl    =data(2,*) 
dl    =data(3,*) 

print,'% Finish loading lens file... ...'
;for i=0,10 do print,ral(i),decl(i),zl(i),dl(i)
;-------------------------------------------------------------

print,'% Start to load the source file... ...'
ns    =file_lines('../lwtsrcs.dat') 
data  =dblarr(7,ns)
openr,lunr,'../lwtsrcs.dat',/get_lun
readf,lunr,data
free_lun,lunr

ras   =data(0,*) & decs  =data(1,*)
zs    =data(2,*) & ds    =data(3,*)
e1    =data(4,*) & e2    =data(5,*)
wt1   =data(6,*)

delvarx,data
print,'% Finish loading the source file... ...'
;--------------------------------------------------------------
;Start to measure shear;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

print,'% Start to create the list... ...'
print,'% Time to start create the list: ',systime(0)

Nmax  =50000
lists =ptrarr(nl,/allocate_heap)

for is=0l,nl-1 do begin
	sep1          =abs(ras-ral(is))
	sep2          =abs(decs-decl(is))
	idx           =where(sep1 le 0.5 and sep2 le 0.5 and zs gt zl(is)+0.1)
	;idx           =where(sep1 le 1.5 and sep2 le 1.5 and zs lt zl(is) and res gt 1.0/3.0,ct)
	;print,format='($%"\r%d")',is
        if n_elements(idx) gt 1 then begin
		*lists(is) =idx
	endif else begin
	        *lists(is) =[0]
	endelse
endfor	

print,'% End of creating the list: ',systime(0)
print,'% Finish creating the list... ...'

;--------------------------------------------------------------

print,'% Calculating the shear... ...'
wgt   =fltarr(Nbin,nl)
countl=fltarr(Nbin)
resp =2.0*(1.0-variance(e1,/nan))
print,'% Starting time: ',systime(0)
	
for j=0l,nl-1 do begin
                
	ipt =*lists(j)
	ra1 =ras[ipt] & dec1=decs[ipt]
	me2=e2[ipt]   & zs1 =zs[ipt]
	dis1=ds[ipt]  & wt=wt1[ipt]

        Sig=fac*dis1*4222.0/(dl(j)*(dis1-dl(j)))/dH0/(1.0+zl(j))/(1.0+zl(j))
        wt =wt/Sig/Sig
	xm0=cos(pi/2.0-dec1*pi/180.)
        xm1=cos(pi/2.0-decl(j)*pi/180.)
        xm2=sin(pi/2.0-dec1*pi/180.)
        xm3=sin(pi/2.0-decl(j)*pi/180.)
        xm4=cos((ra1-ral(j))*pi/180.)
        the=acos(xm0*xm1+xm2*xm3*xm4)

        for iy=0,Nbin-1 do begin

	  dis=dl(j)*the*(1.+zl(j))
	  il =where(dis ge rbin(iy) and dis lt rbin(iy+1),ct2); and finite(rs,/nan) ne 0 ,ct2)
	  if ct2 gt 0 then begin

		wgt(iy,j) =total(wt[il],/nan)
		countl(iy)=countl(iy)+1.0
	  endif
	endfor

endfor
delvarx,ral,decl,zl,dl,ras,decs,zs,ds
delvarx,e1,e2,res,er1,er2,spa
delvarx,lists
print,'% Finish calculating the shear... ...'

;--------------------------------------------------------------
print,'% Start bootstraping... ... '
;
;fcovar=strcompress('for_covar_'+string(1),/remove_all)
;openw,luncov,fcovar,/get_lun
wtf =fltarr(Nsamp)
wt2err =fltarr(Nsamp)

for i=0,Nbin-1 do begin
  wt2=fltarr(Nsamp)
  for j=0,Nsamp-1 do begin
	ibt     =long(randomu(seed,nl)*(nl+1))
	wt2(j)=total(wgt(i,ibt))

       ;printf,luncov,i+1,ga1(j),format='(I3,x,D16.6)'
  endfor
  wt2err(i)=sqrt(variance(wt2))
  wtf(i)   =total(wt2)
;
  ;gamm1(i)=mean(ga1) & g1err(i)=sqrt(variance(ga1))
  ;gamm2(i)=mean(ga2) & g2err(i)=sqrt(variance(ga2))
endfor
;free_lun,luncov
print,'Finish Bootstrap... ...'
;Output;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
output=strcompress('1D_boost_rand',/remove_all)
openw,lunw,output,/get_lun

for ixx=0,Nbin-1 do begin
	printf,lunw,(rbin(ixx)+rbin(ixx+1))/2.0,wtf(ixx),countl(ixx),$
		format='(D16.6,x,D16.6,x,I6)'
endfor
free_lun,lunw

print,'% Finishing time: ',systime(0)
Print,'% Everything is done!'

end
