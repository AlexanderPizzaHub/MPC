import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
from utils import gt


N = 4000
Nb = 1000
Ni = 1000

T = 1
delta = 0.2

dataname =  '5000pts'
data = dict()
data['domain'] = dict() #['0']: main domain, ['1']: domain for pretraining
data['bdry'] = dict()
data['init'] = dict()
data['term'] = dict()

def sample_onepoint(xl,xr,yl,yr,tl,tr):
    x = xl + (xr-xl)*uniform.rvs()
    y = yl + (yr-yl)*uniform.rvs()
    t = tl + uniform.rvs() * (tr-tl)
    return x,y,t

def domain_sampler(size,xl,xr,yl,yr,tl,tr):
    domain_data_x = list()
    domain_data_y = list()
    domain_data_t = list()
    while len(domain_data_x)<size:
        x,y,t = sample_onepoint(xl,xr,yl,yr,tl,tr)
        domain_data_x.append(x)
        domain_data_y.append(y)
        domain_data_t.append(t)

    domain_data = np.array([domain_data_x,domain_data_y,domain_data_t]).T
    print(domain_data.shape)
    return domain_data

#domain_data = domain_sampler(size=N)

data['domain']['0'] = domain_sampler(size=N,xl=0,xr=1,yl=0,yr=1,tl=0,tr=T)
data['domain']['1'] = domain_sampler(size=N/5,xl=0,xr=1,yl=0,yr=1,tl=-delta,tr=0)
#print(domain_data.shape)


def generate_random_bdry(Nb,tl,tr):
    '''
    Generate random boundary points.
    '''
    bdry_col = uniform.rvs(size=Nb*3).reshape([Nb,3])
    for i in range(Nb):
        randind = np.random.randint(0,2)
        if bdry_col[i,randind] <= 0.5:
            bdry_col[i,randind] = 0.0
        else:
            bdry_col[i,randind] = 1.0
        
    bdry_col[:,2] = tl + bdry_col[:,2] * (tr-tl)

    return bdry_col

#bdry_col = generate_random_bdry(Nb)
data['bdry']['0'] = generate_random_bdry(Nb,tl=0,tr=T)

def generate_init(Ni,t):
    '''
    Generate random initial points.
    '''
    init_col = uniform.rvs(size=Ni*3).reshape([Ni,3])

    init_col[:,2] = t + np.zeros_like(init_col[:,2])

    return init_col

#init_col = generate_init(Ni)
data['init']['0'] = generate_init(Ni,t=0)

data['term']['0'] = generate_init(Ni,t=T)

if not os.path.exists('dataset/'):
    os.makedirs('dataset/')

with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(data,pfile)