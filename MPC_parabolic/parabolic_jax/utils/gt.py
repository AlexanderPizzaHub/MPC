from math import sin
import numpy as np
from sympy import *

ld = 0.01

a = 1
T = 1.0

x1,x2,t = symbols('x1 x2, t')

variables = [x1,x2,t]


y_init = sin(np.pi*x1)*sin(np.pi*x2)
y_dat =np.pi**2*x1*(1-x1)*x2*(1-x2)


def from_seq_to_array(items):
    out = list()
    for item in items:
        out.append(np.array(item).reshape(-1,1))
    
    if len(out)==1:
        out = out[0]
    return out

class data_gen():
    def __init__(self):

        self.ydat = lambdify(variables,y_dat,'numpy')
        self.yinit = lambdify(variables,y_init,'numpy')

    def generate(self,ld_exp,col):
        gt =  [
        ld_exp(d[0],d[1],d[2]) for d in col
        ]
        return from_seq_to_array([gt])
   
