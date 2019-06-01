from __future__ import print_function
import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

from scipy import sparse

class uuCF(object):
    def __init__(self, Y_data, k, sim_func = cosine_similarity):
        self.Y_data = Y_data # a 2d array of shape(n_users, 3)
                            # each row of Y_data has form [user_id, item_id, rating]
        self.k = k #number of neighborhood
        self.sim_func = sim_func # similarity function, default: cosine_similarity
        self.Ybar = None # normalize data
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 #number of users
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def fit(self):
        # normalized Y_data -> Ybar
        users = self.Y_data[:, 0] # all users - first column of Y_data
        self.Ybar = self.Y_data.copy()
        self.mu = self.zeros((self.n_users, ))
        for n in xrange(self.n_users):
            #row indices of rating must be user n
            ids = np.where(users == n)[0].astype(np.int32)
            #indices of all items rated by user n 
            item_ids = self.Y_data[ids, 1]
            ratings = self.Y_data[ids, 2]

            #avoid zero devision 
            self.mu[n] = np.mean(ratings) if ids.size > 0 else 0 
            self.Ybar[ids, 2] = ratings - self.mu[n]
        ## from the rating matrix as a sparse matrix 
        self.Ybar = sparse.coo_matrix((self.YBar[:, 2], (self.Ybar[:, 1], self.Ybar[:, 0])), (self.n_items, self.n_users)).tocsr()
        self.S = self.sim_func(self.Ybar.T, self.Ybar.T)

    def pred(self, u, i):
        """ predict the raing of user u for item i"""
        # find item i 
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # all users who rated i 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # similarity of u and users who rated i 
        sim = self.S[u, users_rated_i]
        # most k similar users 
        nns = np.argsort(sim)[-self.k:]
        nearest_s = sim[nns] # and the corresponding similarties
        # the corresponding ratings 
        r = self.Ybar[i, users_rated_i[nns]]
        eps = 1e-8 # a small number to a void zero division
        return (r*nearest_s).sum()/(np.abs(nearest_s).sum() + eps) + self.mu[n]