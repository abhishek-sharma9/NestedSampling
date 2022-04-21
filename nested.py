import copy
import time
import sys
import numpy as np
from numpy import random as rng
import matplotlib.pyplot as plt
import h5py

# How many parameters are there?
num_params = 2

# Load the data
data = np.loadtxt('data.txt')
N = data.shape[0]

jump_sizes = np.array([0.01, 2.])

def from_prior():                      # Taking uniform priors for m and c, m ~ U[0, 1] and c ~ U[0, 10]
    
    m = rng.rand()
    c = 10*rng.rand()

    return np.array([m, c])


def log_prior(params):
    
    m, c = params[0], params[1]
    
    if m < 0. or m > 1.0:
        return -np.Inf
    if c < 0. or c > 10.:
        return -np.Inf
    return 0.


def log_likelihood(params):
    
    m, c = params[0], params[1]
    model = m * data[:,0] + c
    
    return -0.5 * np.sum((data[:,1] - model) ** 2 / data[:,2] ** 2 + np.log(2*np.pi*data[:,2]**2))

def logsumexp(values):
    
    if(values[0] > values[1]):
        result = values[0] + np.log(1+np.exp(values[1]-values[0]))
    else:
        result = values[1] + np.log(1+np.exp(values[0]-values[1]))
    # result = np.log(np.exp(values[0]) + np.exp(values[1]))
    return result


n_live = int(input('No. of live points: '))
steps = int(input('No. of iterations: '))

live_points = np.zeros((n_live,2))
logp = np.zeros(n_live)
logl = np.zeros(n_live)

for i in range(0, n_live): 
    x = from_prior()
    live_points[i,:] = x
    logl[i] = log_likelihood(x)

keep = np.zeros((steps, num_params + 4))

logL = []
logX = [0]
logZ = [-sys.float_info.max]
logw = []

start = time.time()

for i in range(steps):
    
    t1 = time.time()
    worst = int(np.where(logl == logl.min())[0][0])
    logL.append(logl[worst])
    logX.append(-(i+1)/n_live)
    logw.append(np.log(np.exp(-i/n_live) - np.exp(-(i+1)/n_live)))
    values = np.array([logZ[i], logL[i]+logw[i]])
    logZ.append(logsumexp(values))
  
    which = rng.randint(n_live)
    while which == worst:
        which = rng.randint(n_live)
    live_points[worst] = copy.deepcopy(live_points[which])
    
    threshold = copy.deepcopy(logl[worst])
    
    keep[i, :2] = live_points[worst]
    keep[i, 2:6] = logL[i], logX[i+1], logw[i], logZ[i+1]
    

    for j in range(10000):
        
        new = from_prior()
        logp_new = log_prior(new)
        
        # Only evaluate likelihood if prior prob isn't zero [log(prior) is not -infinity].
        logl_new = -np.Inf
        
        if(logp_new != -np.Inf):
            
            logl_new = log_likelihood(new)
            
        if (logl_new >= threshold):
            
            live_points[worst] = new
            logp[worst] = logp_new
            logl[worst] = logl_new
            break 
    
    t2 = time.time()
    print('%d iterations completed......'%(i+1))
    print('time taken per iteration: %f', %(1000*(t2-t1))
    print('time elapsed: %f'%(1000*(t2-start)))
          
file = h5py.File('samples_data_1.hdf5', 'w')
file.create_dataset('keep', data=keep)
file.close()

end = time.time()
print('time taken: {} minutes'.format((end-start)/60.))


