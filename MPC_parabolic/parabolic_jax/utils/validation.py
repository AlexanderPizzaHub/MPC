from curses.ascii import ctrl
import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle as pkl
import json
import seaborn as sns
from . import tools
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
#primal state ground truth
#EX1
'''alpha = 0.01
T = 5.0
def data_gen(collocations):
    y_gt = [
        2*d[2]*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]
        
    u_gt = [
        np.sin(np.pi*d[0]) * np.sin(np.pi*d[1]) * ((T-d[2])**2)/alpha for d in collocations
    ]

    p_gt = [-alpha*u for u in u_gt]

    l = len(u_gt)

    y_data = [
        2*np.pi**2*np.sin(np.pi*d[0])*np.sin(np.pi*d[1])*((T-d[2])**2) + 2*T*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]

    f = [
        (2+4*np.pi**2*collocations[ind,2])*np.sin(np.pi*collocations[ind,0])*np.sin(np.pi*collocations[ind,1]) - u_gt[ind] for ind in range(l)
    ]

    return [y_gt,u_gt,p_gt,y_data,f]'''

#Ex2
'''alpha = 0.01
T = 1.0
def data_gen(collocations):
    y_gt = [
        d[2]*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]
        
    u_gt = [
        np.sin(np.pi*d[0]) * np.sin(np.pi*d[1]) * (-T)*(T-d[2]) for d in collocations
    ]

    p_gt = [-alpha*u for u in u_gt]

    l = len(u_gt)

    y_data = [
        (d[2]-(2*np.pi**2+1)*alpha*T)*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]

    f = [
        (2*np.pi**2*d[2]+1+T*(T-d[2]))*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]

    return [y_gt,u_gt,p_gt,y_data,f]'''

#Ex3
alpha = 0.01
T = 0.3
def data_gen(collocations):
    y_gt = [
        d[2]*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]
        
    u_gt = [
        np.sin(np.pi*d[0]) * np.sin(np.pi*d[1]) * (T-d[2])*(-1.0/alpha) for d in collocations
    ]

    p_gt = [-alpha*u for u in u_gt]

    l = len(u_gt)

    y_data = [
        (d[2]-(2*np.pi**2)*(T-d[2])-1)*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]

    f = [
        (2*np.pi**2*d[2]+1+(T-d[2])/alpha)*np.sin(np.pi*d[0])*np.sin(np.pi*d[1]) for d in collocations
    ]

    return [y_gt,u_gt,p_gt,y_data,f]



'''
Generate the mesh for validation.
'''
val_x=np.arange(0,1,0.04).reshape([25,1])
val_y=np.arange(0,1,0.04).reshape([25,1])
val_t = np.arange(0,T,T/25).reshape([25,1])

val_ms_x,val_ms_y,val_ms_t = np.meshgrid(val_x,val_y,val_t)
val_x = np.ravel(val_ms_x).reshape(-1,1)
val_y = np.ravel(val_ms_y).reshape(-1,1)
val_t = np.ravel(val_ms_t).reshape(-1,1)
tvx,tvy,tvt = tools.from_numpy_to_tensor([val_x,val_y,val_t]) # points used for validation

col = np.concatenate([val_x,val_y,val_t],axis=1)
u_gt_np,ctrl_gt_np,p_gt_np,y_dat_np,f_np = data_gen(col)

u_gt = torch.from_numpy(np.array(u_gt_np).reshape(-1,1))#.float()
ctrl_gt = torch.from_numpy(np.array(ctrl_gt_np).reshape(-1,1))#.float()
p_gt = torch.from_numpy(np.array(p_gt_np).reshape(-1,1))#.float()

#Generate grids to output graph
plot_x=np.arange(0,1,0.02).reshape([50,1])
plot_y=np.arange(0,1,0.02).reshape([50,1])
plot_ms_x, plot_ms_y = np.meshgrid(plot_x, plot_y)
plot_val_x = np.ravel(plot_ms_x).reshape(-1,1)
plot_val_y = np.ravel(plot_ms_y).reshape(-1,1)

t_plot_x = Variable(torch.from_numpy(plot_val_x))#()).to(device)
t_plot_y = Variable(torch.from_numpy(plot_val_y))#()).to(device)
t_plot_t_init = torch.zeros([len(plot_val_x),1]).to(device)




#!!!For drawing functions, everything are two dimentional. Which kind of plot is more suitable?
def plot_2D(net,path):
    pt_u = net(t_plot_x,t_plot_y).detach().numpy().reshape([50,50])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(plot_ms_x,plot_ms_y,pt_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(path)
    plt.close()

def plot_t(net,path,t=0):
    pt_u = net(t_plot_x,t_plot_y,t_plot_t_init+t).detach().numpy().reshape([50,50])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(plot_ms_x,plot_ms_y,pt_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(path)
    plt.close()


#In coupled method, control is obtained by projection
def plot_2D_with_proj(net,projector,path,low,high):
    pt_u = projector(net(t_plot_x,t_plot_y)/(-alpha),low,high).detach().numpy().reshape([50,50])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(plot_ms_x,plot_ms_y,pt_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(path)
    plt.close()


mse_loss = torch.nn.MSELoss()
u_L2 = torch.sqrt(torch.mean(torch.square(u_gt)))
ctrl_L2 = torch.sqrt(torch.mean(torch.square(ctrl_gt)))
p_L2 = torch.sqrt(torch.mean(torch.square(p_gt)))

u_Linf = torch.max(torch.abs(u_gt))
p_Linf = torch.max(torch.abs(p_gt))
ctrl_Linf = torch.max(torch.abs(ctrl_gt))



'''
In coupled method, it solves primal state and adjoint state at the same time. 
Each state contains a boundary loss and pde loss, which will be recorded.
Adding these two loss under given penalty weight, it gives the total loss, alse being recorded.
We should also evaluate the cost, although we do not use it during training.

For validation, the primal state groundtruth and contrl groundtruth will be used.
The validation error has two component: relative L2 and relative L_infinity, and validation grid is on fixed 50*50 grid.
'''
class record_couple(object):
    def __init__(self):
        self.losslist = list()
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()
        self.adjhist = list()
        self.adjres = list()
        self.adjbc = list()
        self.vhist_u = list()
        self.vhist_ctrl = list()
        self.costhist = list()
        self.epoch = 0
        self.vinfu = list()
        self.vinfctrl = list()

    def updateTL(self,loss):
        self.epoch= self.epoch+1
        self.losslist.append(loss)
        

    def updatePL(self,pl,ppde,pbc,al,apde,abc):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        self.adjhist.append(al)
        self.adjres.append(apde)
        self.adjbc.append(abc)
    
    def updateCL(self,cl):
        self.costhist.append(cl)

    def updateVL(self,vl_u,vl_p,vinfu,vinfp):
        self.vhist_u.append(vl_u)
        self.vhist_ctrl.append(vl_p)
        self.vinfu.append(vinfu)
        self.vinfctrl.append(vinfp)
    
    def validate(self,u,ctrl):
        with torch.no_grad():
            uout = u(tvx,tvy,tvt)
            cout = ctrl(tvx,tvy,tvt)
            vu = (torch.sqrt(mse_loss(uout,u_gt))/u_L2).cpu().detach().numpy()
            #vc = (torch.sqrt(mse_loss(ctrl(tvx,tvy,tvt)/(-alpha),ctrl_gt))/ctrl_L2).cpu().detach().numpy()
            vc = (torch.sqrt(mse_loss(cout,p_gt))/p_L2).cpu().detach().numpy()
            vuinf = (torch.max(torch.abs(uout-u_gt))/u_Linf).cpu().detach().numpy()
            vcinf = (torch.max(torch.abs(cout-p_gt))/p_Linf).cpu().detach().numpy()

        self.updateVL(vu,vc,vuinf,vcinf)
        return vu,vc

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.adjhist,self.vhist_u,self.vhist_ctrl]
    
    def plotinfo(self,path):
        plt.subplots(6,figsize=[30,20])
        plt.subplot(231)
        plt.loglog(self.losslist)
        plt.title("total loss")

        plt.subplot(232)
        plt.loglog(self.pdehist)
        plt.loglog(self.adjhist)
        plt.legend(['pde','adj'])
        plt.title("pde/adj loss")

        plt.subplot(233)
        plt.loglog(self.vhist_u)
        plt.loglog(self.vhist_ctrl)
        plt.loglog(self.vinfu)
        plt.loglog(self.vinfctrl)
        plt.legend(['state validation L2','control validation L2','state val Linf','ctrl val Linf'])
        plt.title("validation")

        plt.subplot(234)
        plt.loglog(self.costhist)
        plt.title("cost objective")

        plt.subplot(235)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.title("primal state loss")
        plt.legend(['pde residual','boundary condition'])

        plt.subplot(236)
        plt.loglog(self.adjres)
        plt.loglog(self.adjbc)
        plt.title('adjoint state loss')
        plt.legend(['pde residual','boundary condition'])

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)


'''
In penalty method, it involves no adjoint equations. So there wil be only one boundary/pde record.
But in addition, here will be cost objective.

The validation is same as above.
'''
class record_penalty(object):
    def __init__(self):
        self.losslist = list()
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()
        self.vhist_u = list()
        self.vhist_ctrl = list()
        self.epoch = 0
        self.costhist = list()
        self.best_pinnloss = 9999999.0

    def updateTL(self,loss):
        self.epoch= self.epoch+1
        self.losslist.append(loss)
        

    def updatePL(self,pl,ppde,pbc,cost):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        self.costhist.append(cost)

    def updateVL(self,vl_u,vl_p):
        self.vhist_u.append(vl_u)
        self.vhist_ctrl.append(vl_p)
    
    def validate(self,u,ctrl):
        #In penalty, ctrl itself is NN and do not need projection.
        with torch.no_grad():
            vu = (torch.sqrt(mse_loss(u(tvx,tvy,tvt),u_gt))/u_L2).detach().numpy()
            vc = (torch.sqrt(mse_loss(ctrl(tvx,tvy,tvt),ctrl_gt))/ctrl_L2).detach().numpy()

        self.updateVL(vu,vc)
        return vu,vc

    def validate_u(self,u):
        with torch.no_grad():
            vu = (torch.sqrt(mse_loss(u(tvx,tvy,tvt),u_gt))/u_L2).detach().numpy()
        self.vhist_u.append(vu)
        return vu

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.costhist,self.vhist_u,self.vhist_ctrl]
    
    def plotinfo(self,path):
        plt.subplots(4,figsize=[20,20])
        plt.subplot(221)
        plt.loglog(self.losslist)
        plt.title("total loss")

        plt.subplot(222)
        plt.loglog(self.pdehist)
        plt.loglog(self.costhist)
        plt.legend(['pde loss','cost objective'])
        plt.title("pde loss/cost")

        plt.subplot(223)
        plt.loglog(self.vhist_u)
        plt.loglog(self.vhist_ctrl)
        plt.title("validation")
        plt.legend(['state validation', 'control validation'])

        plt.subplot(224)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.legend(['pde residual','boundary residual'])
        plt.title("pinn loss")

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)


'''
This is used to record information in forward test
'''
class record_forward(object):
    def __init__(self):
        self.losslist = list()
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()
        self.vhist_u = list()
        self.epoch = 0

    def updateTL(self,loss):
        self.epoch= self.epoch+1
        self.losslist.append(loss)
        

    def updatePL(self,pl,ppde,pbc):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)

    def updateVL(self,vl_u):
        self.vhist_u.append(vl_u)
    
    def validate(self,u):
        #In penalty, ctrl itself is NN and do not need projection.
        with torch.no_grad():
            vu = np.sqrt(mse_loss(u(t_vx,t_vy,t_vt),u_gt).detach().numpy())/u_L2

        self.updateVL(vu)
        return vu


    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.vhist_u]
    
    def plotinfo(self,path):
        plt.subplots(4,figsize=[20,20])
        plt.subplot(221)
        plt.loglog(self.losslist)
        plt.title("total loss")

        plt.subplot(222)
        plt.loglog(self.pdehist)
        plt.legend('pde loss')
        plt.title("pde loss")

        plt.subplot(223)
        plt.loglog(self.vhist_u)
        plt.title("validation")
        plt.legend('state validation')

        plt.subplot(224)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.legend(['pde residual','boundary residual'])
        plt.title("pinn loss")

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)



'''
In AONN method, primal state solve and adjoint solve are splitted. Each of them contains pde loss and boundary loss, 
which is be recorded seperately. The record is using individual epoch.

The epoch records the outer loop.

Similar to coupled method, the cost objective will not be used, but still be recorded.

Particularly, in AONN, the control NN is learned from the project GD, hence it is always admissible; there is no total loss.
'''
class record_AONN(object):
    def __init__(self):
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()

        self.adjhist = list()
        self.adjres = list()
        self.adjbc = list()

        self.vhist_u = list()
        self.vhist_ctrl = list()
        self.vinfu = list()
        self.vinfc = list()

        self.costhist = list()
        self.epoch = 0

    def updateEpoch(self):
        self.epoch= self.epoch+1
        

    def updatePL(self,pl,ppde,pbc):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        
    def updateAL(self,al,apde,abc):
        self.adjhist.append(al)
        self.adjres.append(apde)
        self.adjbc.append(abc)

    def updateCL(self,cl):
        self.costhist.append(cl)

    def updateVL(self,vl_u,vl_p,vuinf,vpinf):
        self.vhist_u.append(vl_u)
        self.vhist_ctrl.append(vl_p)
        self.vinfu.append(vuinf)
        self.vinfc.append(vpinf)
    
    def validate(self,u,ctrl):
        with torch.no_grad():
            uout = u(tvx,tvy,tvt)
            cout = ctrl(tvx,tvy,tvt)
            vu = (torch.sqrt(mse_loss(uout,u_gt))/u_L2).cpu().detach().numpy()
            #vc = (torch.sqrt(mse_loss(ctrl(tvx,tvy,tvt)/(-alpha),ctrl_gt))/ctrl_L2).cpu().detach().numpy()
            vc = (torch.sqrt(mse_loss(cout,ctrl_gt))/ctrl_L2).cpu().detach().numpy()
            vuinf = (torch.max(torch.abs(uout-u_gt))/u_Linf).cpu().detach().numpy()
            vcinf = (torch.max(torch.abs(cout-ctrl_gt))/ctrl_Linf).cpu().detach().numpy()

        self.updateVL(vu,vc,vuinf,vcinf)

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.pdehist,self.adjhist,self.vhist_u,self.vhist_ctrl]
    
    def plotinfo(self,path):
        plt.subplots(6,figsize=[30,20])

        plt.subplot(231)
        plt.loglog(self.pdehist)
        plt.title("primal state loss")

        plt.subplot(232)
        plt.loglog(self.adjhist)
        plt.title('adjoint state loss')

        plt.subplot(233)
        plt.loglog(self.vhist_u)
        plt.loglog(self.vhist_ctrl)
        plt.loglog(self.vinfu)
        plt.loglog(self.vinfc)
        plt.legend(['state validation L2','control validation L2','state validation Linf','control validation Linf'])
        plt.title("validation")

        plt.subplot(234)
        plt.loglog(self.costhist)
        plt.title("cost objective")

        plt.subplot(235)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.title("primal state loss")
        plt.legend(['pde residual','boundary condition'])

        plt.subplot(236)
        plt.loglog(self.adjres)
        plt.loglog(self.adjbc)
        plt.title('adjoint state loss')
        plt.legend(['pde residual','boundary condition'])

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)



'''
This is a general info recorder for the experiment.
'''
class expInfo(object):
    def __init__(self):
        self.bestValidation_u = None
        self.bestValidation_c = None
        self.bestTraining = None    #Record the best loss
        self.bestCost = None #record the best cost value
        self.epoch_Termination = None   #if we choose termination upon reaching a given accuracy, it records the number of iterations.
        self.exp_description = None #a text description of this experiment, including comments or basic settings
        self.walltime = None #records the running time

    def saveinfo(self,path):
        info = json.dumps(self.__dict__,indent=4,separators=(',',':'))
        f = open(path,'w')
        f.write(info)
        f.close()
        return True
    

