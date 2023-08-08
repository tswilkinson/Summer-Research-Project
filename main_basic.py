import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from scipy.linalg import solve_triangular
from sklearn.ensemble import RandomForestClassifier
import itertools
from scipy.cluster.vq import kmeans

cor_size = 0.2
jitter = 1e-9
PS_bound = 0.1
num_sims = 200
num_post = 2000
cred = 0.95

d=1
n=500
sig_n = 1.0

def prop_score(x):
    return 0.8 - 1.2*np.linalg.norm(x-np.array([0.5]))

def mreg(x,r):
    return np.linalg.norm(x-np.array([0.6])) + r

X = np.random.uniform(0,1,(n,d))
R = np.asarray([np.random.binomial(1,prop_score(X[i,:]),1) for i in range(n)])
Y = np.asarray([mreg(X[i,:],R[i]) + sig_n*np.random.normal(0,1) for i in range(n)])
Z = np.column_stack((X,R))

RF = RandomForestClassifier().fit(X,np.ravel(R))
prop = RF.predict_proba(X)[:,1]
prop = np.asarray([max(min(prop[i],1.0-PS_bound),PS_bound) for i in range(n)])

Z_BB = np.hstack((np.repeat(X,2,axis=0),np.asarray([0.0,1.0]*n).reshape(2*n,1)))
data_filter = [False]*(2*n)
for i in range(n):
    if R[i] == 0:
        data_filter[2*i] = True
    elif R[i] == 1:
        data_filter[2*i+1] = True

PS_weight = np.zeros(2*n)[:,None]
for i in range(n):
    PS_weight[2*i] = -1.0/(1.0-prop[i])
    PS_weight[2*i+1] = 1.0/prop[i]

M=np.mean(abs(PS_weight[data_filter]))

n_inducing = 50
inducing_variables, _ = kmeans(Z,n_inducing)
first_model = gpflow.models.SGPR(
    (Z,Y),
    kernel=gpflow.kernels.SquaredExponential(lengthscales=[1.0,4.0]),
    inducing_variable=inducing_variables
)

opt = gpflow.optimizers.Scipy()
opt.minimize(first_model.training_loss,first_model.trainable_variables)

gpflow.utilities.print_summary(first_model)

sigma_sq = first_model.parameters.SGPR.likelihood.variance
print(sigma_sq)

cor_var = (cor_size**2)*sigma_sq/((M**2)*n)

class Correction(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(activedims=list(range(d+1))))
    def K(self,X,X2=None):
        if X2 is None:
            X2 = X
        return 
    def K_diag(self,X):
        
        
