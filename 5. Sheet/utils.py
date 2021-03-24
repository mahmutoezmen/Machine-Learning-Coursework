import numpy,scipy.spatial,sklearn,sklearn.datasets
import matplotlib
import matplotlib.pyplot as plt


# ============================================
# DATASETS
# ============================================

# --------------------------------------------
# YEAST classification dataset
# --------------------------------------------
def Yeast():

	# Read the dataset
	f = open("yeast.txt","r").readlines()
	X = numpy.array([l[:-1].strip().split(',') for l in f]) 
	X,T = X[:,1:-1].astype('float'),X[:,-1]
	classes = set(['CYT','NUC','MIT','ME3','ME2','ME1','EXC'])

	# Convert labels to soft indicator vectors
	T = numpy.array([(T==c)*1.0 for c in classes]).T + 1e-2
	T /= T.sum(axis=1)[:,numpy.newaxis]

	# Shuffle the data
	R = numpy.random.mtrand.RandomState(5456).permutation(len(X))
	return X[R],T[R]

# --------------------------------------------    
# BOSTON regression dataset 
# --------------------------------------------
def Housing():

	# Read the dataset
	housing = sklearn.datasets.load_boston()
	X,T = housing.data,housing.target

	# Normalize the labels
	T -= T.mean()

	# Shuffle the data
	R = numpy.random.mtrand.RandomState(5456).permutation(len(X))
	return X[R],T[R]


# ============================================
# PREDICTORS
# ============================================

# --------------------------------------------
# Classifier based on Parzen windows
# --------------------------------------------
class ParzenClassifier:

	def __init__(self,parameter):
		self.parameter = parameter

	def fit(self,X,T):
		self.X,self.T = X*1.0,T*1.0
		self.basescale = scipy.spatial.distance.cdist(self.X,self.X,'sqeuclidean').mean()
		return self

	def predict(self,X):
		D = scipy.spatial.distance.cdist(X,self.X,'sqeuclidean')
		K = -D/(self.basescale*self.parameter)
		K = K - K.max(axis=1)[:,numpy.newaxis]
		K = numpy.exp(K)
		K = K / K.sum(axis=1)[:,numpy.newaxis]
		predictions = numpy.dot(K,self.T)
		predictions /= predictions.sum(axis=1)[:,numpy.newaxis]
		return predictions


# --------------------------------------------
# Regressor based on Parzen windows
# --------------------------------------------
class ParzenRegressor:

	def __init__(self,parameter):
		self.parameter = parameter

	def fit(self,X,T):
		self.X,self.T = X*1.0,T*1.0
		self.basescale = scipy.spatial.distance.cdist(self.X,self.X,'sqeuclidean').mean()
		return self

	def predict(self,X):
		D = scipy.spatial.distance.cdist(X,self.X,'sqeuclidean')
		K = -D/(self.basescale*self.parameter)
		K -= K.max(axis=1)[:,numpy.newaxis]
		K = numpy.exp(K)
		K /= K.sum(axis=1)[:,numpy.newaxis]
		predictions = numpy.dot(K,self.T)
		return predictions



# ============================================
# METHODS FOR SAMPLING
# ============================================

# --------------------------------------------
# Deterministic Sampler
# --------------------------------------------
class Sampler:

	def __init__(self,X,T):

		assert(len(X)==len(T))
		self.X,self.T,self.nbsamples = X*1.0,T*1.0,len(X)
		self.seed = numpy.random.mtrand.RandomState(1234)

	# sample half of the total data
	def sample(self):
		r = self.seed.permutation(self.nbsamples)[:self.nbsamples//2]
		return self.X[r]*1.0,self.T[r]*1.0




# ============================================
# METHODS FOR TESTING
# ============================================

# --------------------------------------------
# Plotting Bias Variance and Error when
# training a certain type of predictor for
# a range of parameters on a dataset.
# --------------------------------------------
def plotBVE(dataset,params,predictortype,getBiasVariance,name):

	X,T = dataset()

	B,V,E = [],[],[]

	for param in params:

		# Method to compute bias and variance
		# (use 50% of the data for sampling and
		# the remaining 50% of the data for testing)

		b,v = getBiasVariance(
			Sampler(X[:len(X)//2],T[:len(X)//2]),
			predictortype(param),
			X[len(X)//2:],
			T[len(X)//2:],
			nbsamples=20)

		B += [b]
		V += [v]
		E += [b+v]

	fig = plt.figure()

	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

	axes.plot(params,E,'-o', c='b', label="Error")
	axes.plot(params,B,'-o', c='r', label="Bias")
	axes.plot(params,V,'-o', c='g', label="Variance")

	axes.set_xlabel('Parameters')
	axes.set_title(name);
	axes.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	axes.set_xscale('log')
	axes.grid(True)

