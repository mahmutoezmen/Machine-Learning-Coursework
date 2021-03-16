import numpy,matplotlib
import matplotlib.pyplot as plt

def render(x,w,h,vmax=0.1):
    x = x.reshape([h,w,62,47])
    z = numpy.ones([h,w,64,49])*vmax
    z[:,:,1:-1,1:-1] = x
    x = z.reshape([h,w,64,49]).transpose([0,2,1,3]).reshape([h*64,w*49])
    plt.figure(figsize=(0.49*2*w,0.64*2*h))
    plt.imshow(x,cmap=plt.cm.gray,vmin=-vmax,vmax=vmax)
    plt.axis('off')
    plt.show()

def scatterplot(x,y,xlabel='',ylabel=''):
    plt.figure(figsize=(3,3))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y,'.')
    plt.show()

