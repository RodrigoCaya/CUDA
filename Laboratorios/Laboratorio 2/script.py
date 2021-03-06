import matplotlib.pyplot as mpimg
import numpy as np
def TXTtoRGB(name):
    RGB = np.loadtxt(name+'.txt', delimiter=' ', skiprows = 1)
    with open(name+'.txt') as imgfile:
        M,N = map(int,imgfile.readline().strip().split())
    img = np.ones((M,N,4))
    for i in range(3):
        img[:,:,i] = RGB[i].reshape((M,N)) 
    mpimg.imsave(name+'_fromfile.png', img)
TXTtoRGB('salidaA')
TXTtoRGB('salidaB')
TXTtoRGB('salidaC')