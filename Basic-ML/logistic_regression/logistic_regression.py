import numpy as np 


def sigmoid(S):
    """
    S: an numpy array
    return sigmoid function of each element of s
    """
    return 1 / (1 + np.exp(-S))
def prob(w, X):
    '''
    X: a 2-d numpy array of shape(N, d) N datapoint, each with size d 
    w: a 1-d numpy array of shape(d)
    '''
    return sigmoid(X.dot(w))

def loss(w, X, y, lam):
    """
    W, x as in prob
    y: a 1-d numpy array of shape(N). Each elem =0 or 1
    """
    z = prob(w, X)
    return -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z)) + 0.5 * lam/X.shape[0] * np.sum(w * w)

def logistic_regression(w_init, X, y, lam = 0.001, lr = 0.1, nepoches = 2000):
    # lam - reg paramether, lr - learning rate, nepochoes number of epoches
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init
    loss_hist = [loss(w_init, X, y, lam)] #store history of loss in loss_hist
    ep = 0
    while ep < nepoches:
        ep += 1
		mix_ids = np.random.permutation(N)
		for i in mix_ids:
			xi = X[i]
			yi = Y[i]
			zi = sigmoid(xi.dot(w))
			w = w - lr*((zi - yi) * xi + lam*w)
		loss_hist.append(loss(w, X, y, lam))
        if np.linalg.norm(w - w_old) / d < 1e-6:
                break
        w_old = w
	return w, loss_hist

def predict(w, X, threshold = 0.5):
	"""
	predict out of each row of x 
	X: a numpy array of shape(N, d)
	threshold: a threshold between 0 and 1 
	return a 1d numpy array, each elemnt is 0 or 1
	"""
	res = np.zeros(X.shape[0])
	res[np.where(prob(w, X)) > threshold)[0]] = 1
	return res