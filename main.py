import numpy as np
import GPy
from scipy.linalg import solve_triangular
from sklearn.linear_model import LogisticRegression

cor_size = 0.2
PS_bound = 0.1
num_post = 2000
cred = 0.95

d=100
n=500
sig_n = 1.0

def prop_score(x):
    if isinstance(x,(list,np.ndarray)):
        if len(x)>=5:
            val = x[0]+(x[1]-0.5)**2 + x[2]**2 - 2*np.sin(2*x[3]) + np.exp(-x[4]) - np.exp(-1.0) + 1.0/6.0
            if val > 0:
                return 1.0
            else:
                return 0.0

def mreg(x,r):
    if isinstance(x,(list,np.ndarray)):
        if len(x)>=5:
            val = np.exp(-x[0])+x[1]**2+x[2]+np.cos(x[4]) + r*(1+2*x[1]*x[4])
            if x[3]>0:
                val = val+1.0
            return val

X = np.random.normal(0,1,(n,d))
R = np.asarray([np.random.binomial(1,prop_score(X[i,:]),1) for i in range(n)])
Y = np.asarray([mreg(X[i,:],R[i]) + sig_n*np.random.normal(0,1) for i in range(n)])
Z = np.column_stack((X,R))

LR = LogisticRegression(solver='lbfgs').fit(X,np.ravel(R))
prop = LR.predict_proba(X)[:,1]
prop = np.asarray([max(min(prop[i],1.0-PS_bound),PS_bound) for i in range(n)])

k = GPy.kern.RBF(d+1,active_dims=list(range(d+1)),name='rbf',ARD=True)
m = GPy.models.GPRegression(Z,Y,k)
m.optimize()
sigma_sq = m.Gaussian_noise.variance

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
PriorCov_observed = PriorCov_BB[data_filter,:][:,data_filter]
(PriorCov_eigvals,PriorCov_eigvecs) = np.linalg.eigh(PriorCov_observed)
print(PriorCov_BB[1:10,1:10])
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

func = lambda l: 1/(l+sigma_sq)
the_diag = np.diag(func(PriorCov_eigvals))
mean_left = np.matmul(K_Lm_n,the_diag)
mean_right = np.matmul(PriorCov_eigvecs.T,Y)

for m in range(50,n+1,50):
    print("m = ",m)
    mean_left_m = mean_left[:,n-m:]
    mean_right_m = mean_right[n-m:]
    K_Lm_m = K_Lm_n[:,n-m:]

    meanLm = np.matmul(mean_left_m,mean_right_m)
    covLm = K_Lm_Lm - np.matmul(mean_left_m,K_Lm_m.T)

    Chol_Lm = np.linalg.cholesky(covLm)
    ATE = np.zeros(num_post)
    for i in range(num_post):
        DP_weights = np.random.exponential(1,n)
        DP_weights = DP_weights/sum(DP_weights)
        GP_draw = meanLm + np.matmul(Chol_Lm,np.random.normal(0,1,(n,1)))
        ATE[i] = np.dot(DP_weights,GP_draw)

    print("mean = ",np.mean(meanLm))
    low = np.quantile(ATE,(1-cred)/2)
    up  = np.quantile(ATE,(1+cred)/2)
    print("up = ",up)
    print("low = ",low)
    print()
