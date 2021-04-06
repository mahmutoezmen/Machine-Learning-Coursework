import numpy,numpy.random,sklearn,matplotlib
from matplotlib import pyplot as plt

def plot_iris(X,T,predict):

    plt.figure(figsize=(4,2.5))

    if predict != None:
        G = numpy.meshgrid(numpy.arange(4,8.1,0.1),numpy.arange(2,4.6,0.1))
        D = numpy.array([G[0].flatten(),G[1].flatten()]).T
        Y = predict(D)
        plt.contourf(G[0],G[1], Y.reshape(G[0].shape), cmap=plt.cm.RdYlBu,alpha=0.4,levels=[-0.5,0.5,1.5,2.5])

    plt.scatter(*X[T==0].T,color='red',s=10)
    plt.scatter(*X[T==1].T,color='orange',s=10)
    plt.scatter(*X[T==2].T,color='blue',s=10)
    
    plt.axis([4,8,2,4.5])
    plt.show()


def split(X,T,seed=0):
    N = len(X)
    P = numpy.random.mtrand.RandomState(seed).permutation(N)
    Xtrain,Xtest = X[P[:N//2]],X[P[N//2:]]
    Ttrain,Ttest = T[P[:N//2]],T[P[N//2:]]
    return (Xtrain,Ttrain),(Xtest,Ttest)


def benchmark(predictor,dataset):
    
    X = dataset.data
    T = dataset.target
    
    nbtrials = 100
    nbclasses = T.max()
    nbsamples = len(X)

    acctr,acctt = 0,0
    
    for i in range(nbtrials):

        (Xtrain,Ttrain),(Xtest,Ttest) = split(X,T,seed=i)
        predictor.fit(Xtrain,Ttrain)
        acctr += predictor.score(Xtrain,Ttrain) / nbtrials
        acctt += predictor.score(Xtest,Ttest) / nbtrials

    return acctr, acctt
