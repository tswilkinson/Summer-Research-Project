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

d=10
n=500
sig_n = 1.0

M_upper = 40

# for each m, keep cumulative sum of absolute error for mean, squared error for mean,
# 95% credible interval width, squared 95% credible interval width.
# Also number of occurances of coverage and of Type II error
ATE_stats = np.zeros((M_upper,6))

ATE_stats_n = np.zeros(6)

x_points = np.zeros((100,10))
x_weights = np.zeros(100)

r_points = np.zeros((100,10))
r_weights = np.zeros(100)

f = open("Points and weights","r")
for i in range(100):
    l = f.readline()
    (p,_,w) = l.partition(";")
    x_strs = p.split(",")
    for j in range(10):
        x_points[i,j] = float(x_strs[j])

    x_weights[i] = float(w)

f.readline()

for i in range(100):
    l = f.readline()
    (p,_,w) = l.partition(";")
    r_strs = p.split(",")
    for j in range(10):
        r_points[i,j] = float(r_strs[j])

    r_weights[i] = float(w)

def prop_score(x):
    y = 0
    for i in range(100):
        y += r_weights[i]*np.linalg.norm(x-r_points[i,:])**0.3
    return 20/(20+y)

def mreg(x,r):
    y = 0
    for i in range(100):
        y += x_weights[i]*np.linalg.norm(x-x_points[i,:])**0.4
    return y + r*(1-0.5*np.cos(2*np.pi*x[0]))

ATE_true = 1.0
print("ATE = ",ATE_true)
print()

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
        if R[i]==0:
            data_filter[2*i] = True
        elif R[i]==1:
            data_filter[2*i+1] = True

    PS_weight = np.zeros(2*n)[:,None]
    for i in range(n):
        PS_weight[2*i] = -1.0/(1.0-prop[i])
        PS_weight[2*i+1] = 1.0/prop[i]

    M=np.mean(abs(PS_weight[data_filter]))
    cor_var = (cor_size**2)*m.rbf.variance/((M**2)*n)

    PriorCov_BB = k.K(Z_BB,Z_BB) + cor_var*np.matmul(PS_weight,PS_weight.T)
    PriorCov_eig = np.real(min(np.linalg.eigvals(PriorCov_BB)))
    print(PriorCov_eig)
    if PriorCov_eig < jitter:
        PriorCov_BB = PriorCov_BB + (jitter - PriorCov_eig)*np.eye(2*n)

    PriorCov_observed = PriorCov_BB[data_filter,:][:,data_filter]
    (PriorCov_eigvals,PriorCov_eigvecs) = np.linalg.eigh(PriorCov_observed)
    print(min(PriorCov_eigvals),max(PriorCov_eigvals))
    print()

    tmp = PriorCov_BB[:,data_filter]
    K_Lm_n = np.zeros((n,n))
    K_Lm_Lm = np.zeros((n,n))
    for i in range(n):
        K_Lm_n[i,:] = tmp[2*i+1,:] - tmp[2*i,:]
        for j in range(i+1):
            K_Lm_Lm[i,j] = PriorCov_BB[2*i+1,2*j+1]-PriorCov_BB[2*i+1,2*j]-PriorCov_BB[2*i,2*j+1]+PriorCov_BB[2*i,2*j]
            K_Lm_Lm[j,i] = K_Lm_Lm[i,j]

    min_eig = np.real(min(np.linalg.eigvals(K_Lm_Lm)))
    if min_eig < jitter:
        K_Lm_Lm = K_Lm_Lm + (jitter - min_eig)*np.eye(n)

    func = lambda l: 1/(l+sigma_sq)
    the_diag = np.diag(func(PriorCov_eigvals))
    K_Lm_ms = np.matmul(K_Lm_n,PriorCov_eigvecs)
    mean_left = np.matmul(K_Lm_ms,the_diag)
    mean_right = np.matmul(PriorCov_eigvecs.T,Y)

    meanLm_n = np.matmul(mean_left,mean_right)
    covLm_n = K_Lm_Lm - np.matmul(mean_left,K_Lm_ms.T)

    Chol_Lm_n = np.linalg.cholesky(covLm_n)
    ATE_n = np.zeros(num_post)
    for i in range(num_post):
        DP_weights = np.random.exponential(1,n)
        DP_weights = DP_weights/sum(DP_weights)
        GP_draw = meanLm_n + np.matmul(Chol_Lm_n,np.random.normal(0,1,(n,1)))
        ATE_n[i] = np.dot(DP_weights,GP_draw)

    meanATE_n = np.mean(meanLm_n)
    sdATE_n = np.std(ATE_n)

    ATE_stats_n[0] += abs(meanATE_n-ATE_true)
    ATE_stats_n[1] += (meanATE_n-ATE_true)**2
    low = np.quantile(ATE_n,(1-cred)/2)
    up  = np.quantile(ATE_n,(1+cred)/2)
    ATE_stats_n[2] += up-low
    ATE_stats_n[3] += (up-low)**2
    if ATE_true >= low and ATE_true <= up:
        ATE_stats_n[4] += 1
    if low <= 0 and up >= 0:
        ATE_stats_n[5] += 1

    for m in range(1,M_upper+1):
        inducing_variables, _ = kmeans(Z,m)
        first_model = gpflow.models.SGPR(
                (Z,Y),
                kernel=gpflow.kernels.SquaredExponential(lengthscales=[1.0 for _ in range(d+1)]),
                inducing_variable=inducing_variables
        )
        
        opt = gpflow.optimizers.Scipy()
        opt.minimize(first_model.training_loss,first_model.trainable_variables)
        
        sigmasq = first_model.parameters.SGPR.likelihood.variance
        print(sigmasq)
        cov_var = (cor_size**2)*sigmasq/((M**2)*n)
        points = np.concatenate(Z_BB,first_model.SGPR.parameters.inducing_variable)
        print(points)
        mus,Ks = first_model.predict_f(points,full_cov=True)
        Kmm = Ks[0,
        mus = musish[:,0]
        Knn = Knnish[0]
        PriorCov_BB = Knn + cov_var*np.matmul(PS_weight,PS_weight.T)
        print(Knn)
        print(mus)
        print(PriorCov_BB)

        Sigma = np.linalg.inv(

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
