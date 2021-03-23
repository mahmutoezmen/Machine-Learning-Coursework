import numpy,matplotlib
from matplotlib import pyplot as plt

class Abalone:

    # Instantiate the Abalone dataset
    #
    # input: None
    # output: None
    #
    def __init__(self):
        X = numpy.genfromtxt('abalone.csv',delimiter=',')
        self.N = X[X[:,0] <2][:,1:-1]
        self.I = X[X[:,0]==2][:,1:-1]

    # Plot a histogram of the projected data
    #
    # input: the projection vector and the name of the projection
    # output: None
    #
    def plot(self,w,name):
        plt.figure(figsize=(6,2))
        plt.xlabel(name)
        plt.ylabel('# examples')
        plt.hist(numpy.dot(self.I,w),bins=25, alpha=0.8,color='red',label='infant')
        plt.hist(numpy.dot(self.N,w),bins=25, alpha=0.8,color='gray',label='noninfant')
        plt.legend()
        plt.show()

