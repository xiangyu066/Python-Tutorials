# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:02:54 2020

@author: XYZ
"""

#%%
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#%% Define functions
def Gaussian2D(xx,yy,amplitude,xc,yc,sigma,offset):
    xc=float(xc)
    yc=float(yc)    
    sigma_x=sigma
    sigma_y=sigma
    theta=0
    a=(np.cos(theta)**2)/(2*sigma_x**2)+(np.sin(theta)**2)/(2*sigma_y**2)
    b=-(np.sin(2*theta))/(4*sigma_x**2)+(np.sin(2*theta))/(4*sigma_y**2)
    c=(np.sin(theta)**2)/(2*sigma_x**2)+(np.cos(theta)**2)/(2*sigma_y**2)
    f=offset+amplitude*np.exp(-(a*((xx-xc)**2)+2*b*(xx-xc)*(yy-yc)+c*((yy-yc)**2)))
    return f

def _Gaussian2D(M,*args):
    x,y=M
    arr=np.zeros(x.shape)
    for i in range(len(args)//5):
       arr+=Gaussian2D(x,y,*args[i*5:i*5+5])
    return arr

#%%
# create meshgrids
x=np.linspace(0,127,128)
y=np.linspace(0,127,128)
xx,yy=np.meshgrid(x,y)

# create virtual data
gprms=[(1,32,23,4,0),
       (1,44,30,4,0),
       (1,100,90,4,0),
       (1,102,94,4,0)]
zz=np.zeros(xx.shape)
for p in gprms:
    zz+=Gaussian2D(xx,yy,*p)
noise=0.05*np.random.normal(size=zz.shape)
zz+=noise

# fitting initiallization
xdata=np.vstack((xx.ravel(), yy.ravel()))
ydata=zz.ravel()

guess_prms=[(1,30,24,4,0),
            (1,46,32,4,0),
            (1,102,91,4,0),
            (1,99,90,4,0)]

p0=[p for prms in guess_prms for p in prms]

# fitting
popt,pcov=optimize.curve_fit(_Gaussian2D,xdata,ydata,p0)
fit=np.zeros(zz.shape)
for i in range(len(popt)//5):
    fit+=Gaussian2D(xx,yy,*popt[i*5:i*5+5])
    
# show fitting result
print('Fitted parameters:')
print(np.reshape(popt,[-1,5]))

rms=np.sqrt(np.mean((zz-fit)**2))
print('RMS residual =', rms)

# Plot the test data as a 2D image and the fit as overlaid contours.
fig,ax=plt.subplots()
ax.imshow(np.flip(zz,0),cmap='plasma',extent=(x.min(),x.max(),y.min(),y.max()))
ax.contour(xx,yy,fit,5,colors='w')
plt.show()