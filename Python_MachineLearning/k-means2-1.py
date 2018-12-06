#https://machinelearningcoban.com/2017/01/04/kmeans2/
import numpy as np
from mnist import MNIST 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from display_network import *

mndata = MNIST('../MNIST/')
mndata.load_testing()
X = mndata.test_images

kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)