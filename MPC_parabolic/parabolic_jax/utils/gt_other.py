import numpy as np
from scipy.fftpack import diff
from sympy import *

alpha = pi**(-4)

a = 1
T = 0.1

t, x1,x2 = symbols('t x1 x2')

w = exp(a*pi**2*t)*sin(pi*x1)*sin(pi*x2)

y = -1.0/(2*t*a) * pi**2 * w
p = exp(a*pi**2*t)*sin(pi*x1)*sin(pi*x2) - exp(a*pi**2*T)*sin(pi*x1)*sin(pi*x2)

y_init = -1.0/(2+a)*pi**2*sin(pi*x1)*sin(pi*x2)

variables = [x1,x2,t]

laplacian_y = 0
for x in variables:
    laplacian_y += diff(y,x,x)
y_t = diff(y,t)

laplacian_p = 0
for x in variables:
    laplacian_p += diff(p,x,x)
p_t = diff(p,t)


u = -p/alpha

#y_dat = y + laplacian_p + p_t
#f = y_t - laplacian_y - u
f = -pi**4 * exp(a*pi**2*T)*sin(pi*x1)*sin(pi*x2)
y_dat = (a**2-5)/(2+a)*pi**2*w + 2*pi**2*exp(a*pi**2*T)*sin(pi*x1)*sin(pi*x2)

#Generates all the ground truth data needed. 

#boundary is periodic, not needed.
ldy = lambdify(variables,y,'numpy')
ldu = lambdify(variables,u,'numpy')
ldp = lambdify(variables,p,'numpy')
ldydat = lambdify(variables,y_dat,'numpy')
ldf = lambdify(variables,f,'numpy')
ldy_init = lambdify(variables,y_init,'numpy')

def from_seq_to_array(items):
    out = list()
    for item in items:
        out.append(np.array(item).reshape(-1,1))
    
    if len(out)==1:
        out = out[0]
    return out

class data_gen():
    def __init__(self):
        self.ldy = lambdify(variables,y,'numpy')
        self.ldu = lambdify(variables,u,'numpy')
        self.ldp = lambdify(variables,p,'numpy')
        self.ldydat = lambdify(variables,y_dat,'numpy')
        self.ldf = lambdify(variables,f,'numpy')

    def data_gen_interior(self,ld_exp,col):
        gt =  [
        ld_exp(d[0],d[1],d[2]) for d in col
        ]
        return from_seq_to_array(gt)
   


def data_gen_interior(collocations):
    #input:(x1,x2,t)
    y_gt = [
        ldy(d[0],d[1],d[2]) for d in collocations
    ]
        
    u_gt = [
        ldu(d[0],d[1],d[2]) for d in collocations
    ]

    p_gt = [
        ldp(d[0],d[1],d[2]) for d in collocations
    ]

    y_data = [
        ldydat(d[0],d[1],d[2]) for d in collocations
    ]
    
    f = [
        ldf(d[0],d[1],d[2]) for d in collocations
    ]

    return from_seq_to_array([y_gt,u_gt,p_gt,y_data,f])

def data_gen_bdry(collocations):
    #how to parse the input?
    y_gt = [
        ldy(d[0],d[1],d[2]) for d in collocations
    ]

    return from_seq_to_array([y_gt])

def data_gen_init(collocations):
    y_gt = [
        ldy_init(d[0],d[1],d[2]) for d in collocations
    ]

    return from_seq_to_array([y_gt])