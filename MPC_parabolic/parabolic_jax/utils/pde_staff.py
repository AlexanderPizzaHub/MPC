import torch

#Function to compute pde term

mse_loss = torch.nn.MSELoss()

def doublegrad(y,x):
    y_x = torch.autograd.grad(y.sum(),x,create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x.sum(),x,create_graph=True)[0]
    
    return y_xx

def pde(x,y,t,net):
    out = net(x,y,t)

    laplace_list = []
    laplace_list.append(doublegrad(out,x))
    laplace_list.append(doublegrad(out,y))

    laplacian = torch.stack(laplace_list,dim=0).sum(dim=0)

    time_grad = torch.autograd.grad(out.sum(),t,create_graph=True)[0]

    return -laplacian + time_grad

def adjoint(x,y,t,net):
    out = net(x,y,t)

    laplace_list = []
    laplace_list.append(doublegrad(out,x))
    laplace_list.append(doublegrad(out,y))

    laplacian = torch.stack(laplace_list,dim=0).sum(dim=0)

    time_grad = torch.autograd.grad(out.sum(),t,create_graph=True)[0]
    return -laplacian - time_grad

#Function to compute the bdry
def compute_state(x,y,t,net):
    out = net(x,y,t)
    return out


#The loss
def pdeloss(net,px,py,pt,pdedata,bx,by,bt,bdrydata,ix,iy,it,initdata,bw,iw):
    
    #pdedata is f.
    pout = pde(px,py,pt,net)
    
    bout = compute_state(bx,by,bt,net)

    iout = compute_state(ix,iy,it,net)
    
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    ires = mse_loss(iout,initdata)
    
    loss = pres + bw*bres + iw*ires

    return loss,[pres,bres,ires],[pdedata-pout,bout-bdrydata,iout-initdata]

def adjloss(net,px,py,pt,pdedata,bx,by,bt,bdrydata,Tx,Ty,Tt,termdata,bw,Tw):
    
    #pdedata is f.
    pout = adjoint(px,py,pt,net)
    
    bout = compute_state(bx,by,bt,net)
    
    Tout = compute_state(Tx,Ty,Tt,net)

    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    Tres = mse_loss(Tout,termdata)

    loss = pres + bw*bres + Tw*Tres
    #loss = pres + bw*bres
    return loss,[pres,bres,Tres],[pdedata-pout,bdrydata-bout,termdata-Tout]

def costfunc(y,data,u,ld,cx,cy,nx,ny):
    #evaluate by trapezoidal

    yout = y(cx,cy)
  
    yre = yout.reshape([nx,ny])
    
    if not isinstance(u,torch.Tensor):
        uout = u(cx,cy)
    else:
        uout = u
    ure = uout.reshape([nx,ny])

    dre = data.reshape([nx,ny])
    
    misfit = 0.5 *torch.square(yre-dre) + ld * 0.5 * torch.square(ure)

    cost = torch.trapezoid(
        torch.trapezoid(misfit,dx=1/(nx-1)),
        dx=1/(ny-1)
        )
    
    return cost

def cost_mse(y,data,u,ld,cx,cy,ct,volume):
    #Estimate of integral: volume*mean
    yout = y(cx,cy,ct)
    if not isinstance(u,torch.Tensor):
        uout = u(cx,cy,ct)
    else:
        uout = u
    misfit = 0.5 *torch.square(yout-data) + ld * 0.5 * torch.square(uout)
    cost = torch.mean(misfit) * volume
    return cost
