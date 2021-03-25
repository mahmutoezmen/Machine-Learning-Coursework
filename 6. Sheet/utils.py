import numpy
import matplotlib
from matplotlib import pyplot as plt


def getdata():

    X = numpy.concatenate([
        numpy.random.normal(0,1,[10000,1000])+numpy.random.randint(-1,2,[1000])*0.2,
        numpy.random.normal(0,1,[10000,1000])+numpy.random.randint(-1,2,[1000])*0.2,
    ])

    T = numpy.concatenate([
        numpy.zeros([10000]),
        numpy.ones([10000]),
    ])

    return X,T


def getcurves(X,T,vals,varv,geterr,hoeffding,vc):
    
    etrains,etests,hoefbounds,vcbounds = [],[],[],[]

    for N,d in vals:

        etrain,etest = geterr(N,d,X,T)

        etrains    += [etrain]
        etests     += [etest]
        hoefbounds += [hoeffding(etrain,N)]
        vcbounds   += [vc(etrain,N,d)]

    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(varv,hoefbounds,label='hoeff',color='red')
    plt.plot(varv,vcbounds,label='vc',color='blue')
    plt.plot(varv,etrains,label='train',color='black',ls='dotted')
    plt.plot(varv,etests,label='test',color='black')
    plt.legend()
    plt.grid(True)
    plt.show()

