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
num_sims = 10
num_post = 2000
cred = 0.95

d=4
n=500
sig_n = 1.0

M_upper = 10

ATE_stats = np.zeros((M_upper,6))
ATE_stats_n = np.zeros(6)

def prop_score(x):
    return 0.8 - 0.6*np.linalg.norm(x-np.array([0.5,0.5,0.5,0.5]))

def mreg(x,r):
    return np.linalg.norm(x-np.array([0.6,0.6,0.6,0.6]))**0.6 + r*(1-0.5*np.cos(2*np.pi*x[0]))

ATE_true = 1.0

class Correction(gpflow.kernels.Kernel):
    def __init__(self,cv,rf):
        super().__init__()
        self.cv = cv
        self.rf = rf
    def K(self,ZZ,ZZ2=None):
        if ZZ2 is None:
            ZZ2 = ZZ
        XX = ZZ[:,:d]
        XX2 = ZZ2[:,:d]
        RR = ZZ[:,d]
        RR2 = ZZ2[:,d]

        pihats = self.rf.predict_proba(XX)[:,1]
        pihats = np.asarray([max(min(ph,1.0-PS_bound),PS_bound) for ph in pihats])
        pihats2 = self.rf.predict_proba(XX2)[:,1]
        pihats2 = np.asarray([max(min(ph,1.0-PS_bound),PS_bound) for ph in pihats2])
        
        brackets = np.asarray([RR[i]/pihats[i] - (1-RR[i])/(1-pihats[i]) for i in range(len(XX))])
        brackets2 = np.asarray([RR2[i]/pihats2[i] - (1-RR2[i])/(1-pihats2[i]) for i in range(len(XX2))])
        return self.cv*np.outer(brackets,brackets2)

    def K_diag(self,ZZ):
        XX = ZZ[:,:d]
        RR = ZZ[:,d]

        pihats = self.rf.predict_proba(XX)[:,1]
        pihats = np.asarray([max(min(ph,1.0-PS_bound),PS_bound) for ph in pihats])

        brackets = np.asarray([RR[i]/pihats[i] - (1-RR[i])/(1-pihats[i]) for i in range(len(XX))])

        return self.cv*np.square(brackets)


for run in range(num_sims):
    print("run ",run)

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

    first_model_n = gpflow.models.GPR(
        (Z,Y),
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0]*(d+1))
    )

    opt_n = gpflow.optimizers.Scipy()
    opt_n.minimize(first_model_n.training_loss,first_model_n.trainable_variables)

    gpflow.utilities.print_summary(first_model_n)

    olengthscales_n = first_model_n.kernel.lengthscales
    okvariance_n = first_model_n.kernel.variance
    olvariance_n = first_model_n.likelihood.variance

    cor_var_n = (cor_size**2)*olvariance_n/((M**2)*n)

    second_model_n = gpflow.models.GPR(
        (Z,Y),
        kernel=gpflow.kernels.SquaredExponential(lengthscales=olengthscales_n,variance=okvariance_n)+Correction(cv=cor_var_n,rf=RF)
    )
    second_model_n.likelihood.variance.assign(olvariance_n)

    (PosteriorMean_BB_n,PosteriorCov_BB_n) = second_model_n.predict_f(np.array(Z_BB),full_cov=True)
    PosteriorMean_BB_n = PosteriorMean_BB_n[:,0]
    PosteriorCov_BB_n = PosteriorCov_BB_n[0,:,:]

    meanLm_n = np.zeros((n,1))
    covLm_n = np.zeros((n,n))

    for i in range(n):
        meanLm_n[i] = PosteriorMean_BB_n[2*i+1] - PosteriorMean_BB_n[2*i]
        for j in range(i+1):
            covLm_n[i,j] = (PosteriorCov_BB_n[2*i+1,2*j+1]-PosteriorCov_BB_n[2*i+1,2*j]
                                 -PosteriorCov_BB_n[2*i,2*j+1]+PosteriorCov_BB_n[2*i,2*j])
            covLm_n[j,i] = covLm_n[i,j]

    min_eig_n = np.real(min(np.linalg.eigvals(covLm_n)))
    if min_eig_n < jitter:
        covLm_n = covLm_n + (jitter-min_eig_n)*np.eye(n)

    Chol_Lm_n = np.linalg.cholesky(covLm_n)
    ATE_n = np.zeros(num_post)
    for i in range(num_post):
        DP_weights = np.random.exponential(1,n)
        DP_weights = DP_weights/sum(DP_weights)
        GP_draw = meanLm_n + np.matmul(Chol_Lm_n,np.random.normal(0,1,(n,1)))
        ATE_n[i] = np.dot(DP_weights,GP_draw)

    ATE_stats_n[0] += abs(np.mean(meanLm_n)-ATE_true)
    ATE_stats_n[1] += (np.mean(meanLm_n)-ATE_true)**2
    low = np.quantile(ATE_n,(1-cred)/2)
    up  = np.quantile(ATE_n,(1+cred)/2)
    ATE_stats_n[2] += up-low
    ATE_stats_n[3] += (up-low)**2
    if ATE_true >= low and ATE_true <= up:
        ATE_stats_n[4] += 1
    if low <= 0 and up >= 0:
        ATE_stast_n[5] += 1

    for m in range(1,M_upper+1):
        inducing_variables, _ = kmeans(Z,m)
        first_model = gpflow.models.SGPR(
            (Z,Y),
            kernel=gpflow.kernels.SquaredExponential(lengthscales=[1.0]*(d+1)),
            inducing_variable=inducing_variables
        )

        opt = gpflow.optimizers.Scipy()
        opt.minimize(first_model.training_loss,first_model.trainable_variables)

#        gpflow.utilities.print_summary(first_model)

        oinducing_points = first_model.inducing_variable.Z
        olengthscales = first_model.kernel.lengthscales
        okvariance = first_model.kernel.variance
        olvariance = first_model.likelihood.variance
        #(oinducing_points,olengthscales,okvariance,olvariance) = first_model.parameters
        #print(olvariance)

        cor_var = (cor_size**2)*olvariance/((M**2)*n)

        second_model = gpflow.models.SGPR(
            (Z,Y),
            kernel=gpflow.kernels.SquaredExponential(lengthscales=olengthscales,variance=okvariance)+Correction(cv=cor_var,rf=RF),
            inducing_variable=oinducing_points
        )
        second_model.likelihood.variance.assign(olvariance)

        (PosteriorMean_BB,PosteriorCov_BB) = second_model.predict_f(np.array(Z_BB),full_cov=True)
        PosteriorMean_BB = PosteriorMean_BB[:,0]
        PosteriorCov_BB = PosteriorCov_BB[0,:,:]

        meanLm = np.zeros((n,1))
        covLm = np.zeros((n,n))

        for i in range(n):
            meanLm[i] = PosteriorMean_BB[2*i+1] - PosteriorMean_BB[2*i]
            for j in range(i+1):
                covLm[i,j] = (PosteriorCov_BB[2*i+1,2*j+1]-PosteriorCov_BB[2*i+1,2*j]
                                   -PosteriorCov_BB[2*i,2*j+1]+PosteriorCov_BB[2*i,2*j])
                covLm[j,i] = covLm[i,j]

        min_eig = np.real(min(np.linalg.eigvals(covLm)))
        if min_eig < jitter:
            covLm = covLm + (jitter-min_eig)*np.eye(n)

        Chol_Lm = np.linalg.cholesky(covLm)
        ATE = np.zeros(num_post)
        for i in range(num_post):
            DP_weights = np.random.exponential(1,n)
            DP_weights = DP_weights/sum(DP_weights)
            GP_draw = meanLm + np.matmul(Chol_Lm,np.random.normal(0,1,(n,1)))
            ATE[i] = np.dot(DP_weights,GP_draw)

        ATE_stats[m-1,0] += abs(np.mean(meanLm)-ATE_true)
        ATE_stats[m-1,1] += (np.mean(meanLm)-ATE_true)**2
        low = np.quantile(ATE,(1-cred)/2)
        up  = np.quantile(ATE,(1+cred)/2)
        ATE_stats[m-1,2] += up-low
        ATE_stats[m-1,3] += (up-low)**2
        if ATE_true >= low and ATE_true <= up:
            ATE_stats[m-1,4] += 1
        if low <= 0 and up >= 0:
            ATE_stats[m-1,5] += 1

print("full posterior:")
mean_mean = ATE_stats_n[0]/num_sims
mean_standard_deviation = np.sqrt(ATE_stats_n[1]/num_sims-mean_mean**2)
print("Average absolute error of posterior mean: {} plus minus {}".format(mean_mean,mean_standard_deviation))
width_mean = ATE_stats_n[2]/num_sims
width_standard_deviation = np.sqrt(ATE_stats_n[3]/num_sims-width_mean**2)
print("Average CI size: {} plus/min {}".format(width_mean,width_standard_deviation))
print("Average coverage: {} Average Type II error: {}".format(ATE_stats_n[4]/num_sims,ATE_stats_n[5]/num_sims))
print()

for m in range(1,M_upper+1):
    print("m = ",m)
    mean_mean = ATE_stats[m-1,0]/num_sims
    mean_standard_deviation = np.sqrt(ATE_stats[m-1,1]/num_sims-mean_mean**2)
    print("Average absolute error of posterior mean: {} plus minus {}".format(mean_mean,mean_standard_deviation))
    width_mean = ATE_stats[m-1,2]/num_sims
    width_standard_deviation = np.sqrt(ATE_stats[m-1,3]/num_sims-width_mean**2)
    print("Average CI size: {} plus/min {}".format(width_mean,width_standard_deviation))
    print("Average coverage: {} Average Type II error: {}".format(ATE_stats[m-1,4]/num_sims,ATE_stats[m-1,5]/num_sims))
    print()
