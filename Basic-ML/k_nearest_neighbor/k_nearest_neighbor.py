from __future__ import print_function
import numpy as np 
from time import time

d, N = 1000, 10000
X = np.random.rand(N, d)
z = np.random.rand(d)

# naively compute square distance between two vector
def dist_pp(z, x):
    d = z - x.reshape(z.shape)
    return np.sum(d * d)
# from one point to each point in a set, value 
def dist_ps_naive(z, X):
	N = X.shape[0]
	res = np.zeros((1, N))
	for i in range(N):
		res[0][i] = dist_pp(z, X[i])

	return res

# from onepoint to each point in a set, fast
def dist_ps_fast(z, X):
	X2 = np.sum(X*X, 1)
	z2 = np.sum(z*z)
	return X2 + z2 - 2 * X.dot(z)

t1 = time()
D1 = dist_ps_naive(z, X)
print('naive point2set, running time:', time() - t1, 's')

t1 = time()
D2 = dist_ps_fast(z, X)
print("fast point2set, running time:", time() - t1, 's');
print("result difference:", np.linalg.norm(D1 - D2))


M = 100
z = np.random.rand(M, d)

#from each point in one set to each point in another set, half fast
def dist_ss_0(Z, X):
	M = Z.shape[0]
	N - X.shape[0]
	res = np.zeros((M, N))
	for i in range(M):
		res[i] = dist_ps_fast(Z[i], x)
	return res

def dist_ss_fast(Z, X):
	X2 = np.sum(X*X, 1)
	Z2 = np.sum(Z*Z, 1)
	return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2 * Z.dot(X.T)

