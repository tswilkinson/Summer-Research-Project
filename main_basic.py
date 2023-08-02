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

n_inducing = 50
inducing_variables, _ = kmeans(Z,n_inducing)
first_model = gpflow.models.SGPR(
    (Z,Y),
    kernel=gpflow.kernels.SquaredExponential(lengthscales=[1.0,1.0]),
    inducing_variable=inducing_variables
)

opt = gpflow.optimizers.Scipy()
opt.minimize(first_model.training_loss,first_model.trainable_variables)

gpflow.utilities.print_summary(first_model)


