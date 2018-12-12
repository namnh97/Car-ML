# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 21:13:17 2018

@author: namnh997
"""

import numpy as np 

#Momentum
#check convergence
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new)) / len(theta_new) < 1e-3

def GD_momentum(theta_init, grad, eta, gamma):
    #suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma * v_old + eta*grad[theta[-1]]
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break
        theta.append(theta_new)
        v_old = v_new
    return theta
    #this vairable includes all points in the path
    #if you just want the final answer,
    #user 'return theta[-1]'


    