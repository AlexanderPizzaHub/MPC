import torch
from torch.autograd import Variable

#This version allows training multiple neural networks withing PDE.
#gpu version still debugging
torch.set_default_dtype(torch.float64)
def from_numpy_to_tensor_with_grad(numpys,device='cpu'):
    #numpys: a list of numpy arrays.
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=True).to(device)
        )

    return outputs

def from_numpy_to_tensor(numpys,device='cpu'):
    #numpys: a list of numpy arrays.
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=False).to(device)
        )

    return outputs

def checkgrad(listoftensors):
    for ts in listoftensors:
        if ts.grad is not None:
            ts.grad.zero_()