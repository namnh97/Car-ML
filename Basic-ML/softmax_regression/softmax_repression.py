import numpy as np
def softmax(Z):
    """
    compute softmax values for each sets of scores in V
    each column of V is a set of scores.
    Z: a numpy array of shape(N, C)
    return a numpy array of shape(N, C)
    """
    e_Z = np.exp(Z) 
    A = e_Z / e_Z.sum(axis=1, keepdims=True)
    return A

def softmax_stable(Z):
    """ Compute softmax values for each sets of scores in Z.
    each row of Z is a set of scores.
    """
    # Z = Z.reshape(Z.shape[0], -1)
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = e_Z / e_Z.sum(axis=1, keepdims=True)
    return A

def softmax_loss(X, y, W):
    """
    W: 2d numpy array of shape(d, C),
        each column corresponding to one ouput node
    x: 2d numpy array of shape(N, d), each row is one data point
    y: 1d numpy array -- label ofe each row of X
    """
    A = softmax_stable(X.dot(W))
    id0 = range(X.shape[0])
    return -np.mean(np.log(A[id0, y]))


def softmax_grad(X, y, W):
    """
    W: 2d numpy array of shape(d, C)
    x: 2d numpy array of shape(N, d) each row is one data point 
    y: 1d numpy array --label of each row of X
    """
    A = softmax_stable(X.dot(W)) #  shape of (N, C)
    id0 = range(X.shape[0])
    A[id0, y] -= 1  # A - Y, shape of (N, C)
    return X.T.dot(A)/X.shape[0]

def softmax_fit(X, y, W, lr=0.01, nepoches=100, tol=1e-5, batch_size=10):
    W_old = W.copy()
    ep = 0
    loss_hist = [softmax_loss(X, y, W)] # store history of loss
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))
    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N)
        for i in range(nbatches):
            # get the i-th batch
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= lr*softmax_grad(X_batch, y_batch, W) #update gradient descent
        loss_hist.append(softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old) / W.size < tol:
            break
        W_old = W.copy()
    return W, loss_hist

def pred(W, X):
    """
    predict out of each columns of X. Class of each x_I is determine by 
    location of max probability. Note that classes are index from 0.
    """
    return np.argmax(X.dot(W), axis=1)
