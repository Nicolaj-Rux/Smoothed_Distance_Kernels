# LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib jupyter-lab
import torch
#import torchvision
import matplotlib.pyplot as plt
from simple_torch_NFFT import Fastsum
import ot
import numpy as np
import scipy
import time
#from IPython import display
import h5py
import pykeops

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
#torch.manual_seed(8)

import torchvision.datasets as td
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

mnist = td.MNIST("mnist", transform=transforms.ToTensor(), download=True)
batch_size = 100
data = DataLoader(dataset=mnist, batch_size=batch_size)
y = next(iter(data))[0].view(batch_size, -1)
label = next(iter(data))[1]
#y = y[label == 4, :]
print(y.shape)

y = y.view(len(y), 28, 28)
#X = torch.cat((X,torch.rot90(X,dims=[-2,-1]),torch.rot90(X,k=2,dims=[-2,-1]) ,torch.rot90(X,k=3,dims=[-2,-1])),0)
#X = torch.cat((X, X.transpose(-2, -1)), 0).view(-1, 28**2)

fig, ax = plt.subplots(4,10, figsize=[10, 4])
ax = ax.reshape(-1)
for i in range(min(len(ax),len(y))):
    im = ax[i].imshow(y[i,:].detach().cpu().reshape(28,28))
    ax[i].axis('off')

y = y.view(-1, 28**2)
y = y.detach()

weights = torch.ones((len(y),), device=device) / len(y)
y = y.to(device=device)

d = y.shape[1] # data dimension
kernel = "energy" # kernel type
fastsum = Fastsum(d, kernel=kernel, device=device, slicing_mode="iid", batched_autodiff=True, 
                    batch_size_nfft=1280, batch_size_P=1280) # fastsum object
scale = 1.0 # kernel parameter

N = len(y) # Number of points in x

# Quadrature points on the sphere
#f1 = h5py.File("distance_directions/d784/P_sym5120.h5","r")
f1 = h5py.File("distance_directions/d784/P_sym2560.h5")
xi0 = torch.tensor(f1['xis'], device=device, dtype=torch.float)
#xi0 = torch.eye(d, device=device, dtype=torch.float)

xi0 = torch.cat(( # Simplex
        (1+1/d)**.5 * torch.eye(d, device=device, dtype=torch.float)
        - ((d+1)**.5+1) * torch.ones((d,d), device=device, dtype=torch.float) / d**1.5,
        torch.ones((1,d), device=device, dtype=torch.float) / d**.5 ))

#xi0 = torch.randn((500,d), device=device, dtype=torch.float) # uniform
#xi0 = xi0 / (xi0**2).sum(1,keepdim=True).sqrt()

def rotate_points(xi):
    # random rotations for obtaining an unbiased estimator:
    d = xi.shape[-1]
    rot = torch.randn((d, d), device=device)
    rot, _ = torch.linalg.qr(rot)
    return torch.matmul(xi0, rot)

# Algorithm parameters
stepsize = 1 
numit_log = 15
numit = 2**numit_log + 1
momentum = 0
k_pow2 = np.unique(np.round(2**np.linspace(0,numit_log, 10*numit_log+1)).astype(int)) #2**np.arange(numit_log+1) # for plot

torch.manual_seed(8)
x = torch.rand((N, d), device=device, dtype=torch.float) 
x.requires_grad_(True)
loss_vector = torch.zeros((numit,))
W2_vector = torch.zeros((numit,))
v = torch.zeros_like(x)
print(f'N = {N}  momentum = {momentum:.4g}  P = {len(xi0)}')

def loss(xis):
    return torch.sum(-fastsum(y, x, weights, scale, xis) + 0.5* fastsum(x, x, weights, scale, xis))
    #return torch.sum(-fastsum.naive(y, x, weights, scale) + 0.5* fastsum.naive(x, x, weights, scale))
def loss_0(xis):
    return torch.sum(0.5 *fastsum(y, y, weights, scale, xis))

tic = time.perf_counter()
x_saved = x.detach().cpu()[:,:,None]
for epoch in range(numit):
    xis = rotate_points(xi0)
    l = loss(xis)
    l.backward()
    with torch.no_grad():
        v = x.grad + momentum * v
        x -= stepsize * v
    x.grad.zero_()
    loss_vector[epoch] = (2 * (l.item() + loss_0(xis)) / N).sqrt()
    W2_vector[epoch] = ot.emd2(weights, weights, ot.dist(x.detach(),y.detach())).sqrt()
    if (epoch & (epoch-1) == 0): #epoch % 100 == 0:
        epoch_log = epoch.bit_length() - 1
        print(f'epoch {epoch}: loss = {loss_vector[epoch]:.5f}  W2 = {W2_vector[epoch]:.5f}  time = {time.perf_counter() - tic:.2f}') #l.item()
        x_saved = torch.cat((x_saved, x.detach().cpu()[:,:,None]), dim=2)

num_figs = x_saved.shape[2] - 6
fig, ax = plt.subplots(num_figs,10, figsize=[10, num_figs])
for i in range(10):
    for j in range(num_figs):
        im = ax[j,i].imshow(x_saved[i,:,j + 6].detach().cpu().reshape(28,28))
        ax[j,i].axis('off')   
fig.savefig(f"out/mnist_N{N}_it{epoch:06d}_m{momentum:.4g}_P{len(xi0)}_result.png", bbox_inches='tight', pad_inches=0.0) 
plt.close(fig)
torch.cuda.empty_cache()

#fig, ax = plt.subplots()
#ax.loglog(loss_vector[10:], W2_vector[10:])
#fig.savefig(f"out/mnist_N{N}_it{numit}_m{momentum:.6g}_P{len(xi0)}_loss.png", bbox_inches='tight', pad_inches=0.0)
np.savetxt(f'out/mnist_N{N}_it{numit}_m{momentum:.6g}_P{len(xi0)}_loss.txt', 
           torch.stack((torch.tensor(k_pow2),loss_vector[k_pow2],W2_vector[k_pow2]),dim=1).numpy(),
           fmt='%.6g')



# Smoothed Riesz
factor_C = torch.tensor(np.exp(scipy.special.loggamma(d/2) - scipy.special.loggamma((1+d)/2)) / np.sqrt(np.pi), 
                        device=device, dtype=torch.float)
print(f'factor_ C = {factor_C}')
def slice_sum_keops(x, y, scale, xi, fun):
    P_local = xi.shape[0]
    x_proj = (x @ xi.T)[:, None, :]
    y_proj = (y @ xi.T)[None, :, :]
    y_proj = pykeops.torch.LazyTensor(y_proj)
    x_proj = pykeops.torch.LazyTensor(x_proj)
    kernel_sum = fun((x_proj - y_proj).abs(), scale).sum(0) / P_local / len(x)
    return kernel_sum

for eps in [.1, .01, .001, .0001, 1]:
    torch.manual_seed(8)
    x = torch.rand((N, d), device=device, dtype=torch.float) 
    x.requires_grad_(True)
    weights = torch.ones((N,), device=device) / N
    
    #eps = torch.tensor(eps, device=device, dtype=x.dtype) # kernel parameter epsilon
    print(f'Smoothed Riesz eps = {eps:.4g}')
    
    def smooth_rest_f(t, scale):
        return torch.where(torch.abs(t/scale) > eps, 0, (eps/3 * (-torch.abs(eps*t/scale)**3 + 3*(eps*t/scale)**2 + 1) 
                           - torch.abs(t/scale)))
    def smooth_f(t, scale):
        return torch.where((t/scale) > eps, t/scale, eps/3 * (-(eps*t/scale)**3 + 3*(eps*t/scale)**2 + 1) )
    def smooth_f_keops(t, scale):
        return (t/scale - eps).ifelse(t/scale, eps/3 * ((eps*t/scale)*(-(eps*t/scale) + 3) + 1))
    
    def smooth_rest_ft(t, scale):
        return torch.nan_to_num((-1 + 2*torch.pi**2 * eps**2 * scale**2 * t*t + torch.cos(2 *torch.pi * eps * scale * t))/
                            (4*torch.pi**4 * eps**2 * scale**3 * t**4) / factor_C,  nan=0,posinf=0,neginf=0)

    def slice_sum(xx, yy, scale, xi):
        P_local = xi.shape[0]
        x_proj = (xx @ xi.T).reshape(-1, 1, P_local)
        y_proj = (yy @ xi.T).reshape(1, -1, P_local)
        kernel_sum = torch.sum(smooth_f((x_proj - y_proj).abs(), scale)) / P_local / len(xx)
        return kernel_sum
    
    #fastsum2 = Fastsum(d, kernel="other", device=device, slicing_mode="orthogonal", 
    #                   kernel_params = {"basis_f": smooth_rest_f}) # fastsum object 
    #                    # "basis_f": smooth_rest_f, "fourier_fun": smooth_rest_ft
    #fastsum3 = Fastsum(d, kernel="other", device=device, slicing_mode="orthogonal", kernel_params = {"fourier_fun": smooth_rest_ft})
    scale = 1.0
    
    def loss2keops(xis):
        return -(-slice_sum_keops(y, x, scale, xis,smooth_f_keops) 
                 + 0.5* slice_sum_keops(x, x, scale, xis,smooth_f_keops)) / factor_C
    def loss2naive(xi):
        return -(-slice_sum(y, x, scale, xi) + 0.5* slice_sum(x, x, scale, xi)) / factor_C
    def loss2naive_0(xi):
        return -(0.5* slice_sum(y, y, scale, xi)) / factor_C
    
    tic = time.perf_counter()
    loss_vector = torch.zeros((numit,))
    W2_vector = torch.zeros((numit,))
    v = torch.zeros_like(x)
    x_saved = x.detach().cpu()[:,:,None]
    
    for epoch in range(numit):
        xis = rotate_points(xi0)
        l = loss2naive(xis)
        l.backward()
        with torch.no_grad():
            v = x.grad + momentum * v
            x -= stepsize * v
        x.grad.zero_()
        loss_vector[epoch] = (2 * (l.item() + loss2naive_0(xis)) / N).sqrt()
        W2_vector[epoch] = ot.emd2(weights, weights, ot.dist(x.detach(),y.detach())).sqrt()
        
        if (epoch & (epoch-1) == 0):
            x_saved = torch.cat((x_saved, x.detach().cpu()[:,:,None]), dim=2)
            epoch_log = epoch.bit_length() - 1
            print(f'epoch {epoch}: loss = {loss_vector[epoch]:.5f}  W2 = {W2_vector[epoch]:.5f}  time = {time.perf_counter() - tic:.2f}') #l.item()

            #fig, ax = plt.subplots(1,10, figsize=[10, 1])
            #ax = ax.reshape(-1)
            #for i in range(len(ax)):
            #    im = ax[i].imshow(x[i,:].detach().cpu().reshape(28,28))
            #    ax[i].axis('off')   
            #fig.savefig(f"out/mnist_N{N}_smootheps{eps:.4g}_it{epoch:06d}_m{momentum:.4g}_P{len(xi0)}_result.png", bbox_inches='tight', pad_inches=0.0) 
            #plt.close(fig)
    
    num_figs = x_saved.shape[2] - 6
    fig, ax = plt.subplots(num_figs,10, figsize=[10, num_figs])
    for i in range(10):
        for j in range(num_figs):
            im = ax[j,i].imshow(x_saved[i,:,j + 6].detach().cpu().reshape(28,28))
            ax[j,i].axis('off')   
    fig.savefig(f"out/mnist_N{N}_smootheps{eps:.04g}_it{epoch:06d}_m{momentum:.4g}_P{len(xi0)}_result.png", bbox_inches='tight', pad_inches=0.0) 
    
    torch.cuda.empty_cache()
    print(f'time {time.perf_counter() - tic}')
    #fig, ax = plt.subplots()
    #ax.loglog(loss_vector[10:], W2_vector[10:])
    #fig.savefig(f"out/mnist_N{N}_smootheps{eps:.04g}_it{numit}_m{momentum:.6g}_P{len(xi0)}_loss.png", bbox_inches='tight', pad_inches=0.0)
    np.savetxt(f'out/mnist_N{N}_smootheps{eps:.04g}_it{numit}_m{momentum:.6g}_P{len(xi0)}_loss.txt', 
               torch.stack((torch.tensor(k_pow2),loss_vector[k_pow2],W2_vector[k_pow2]),dim=1).numpy(),
               fmt='%.6g')
    
    
    # Sinkhorn
    #M = ot.dist(x.detach().cpu(),y.cpu())
    #print(f"Smooth Riesz eps = {eps:.5g}")
    ##print(f"W2 = {torch.tensordot(ot.sinkhorn(weights_x.cpu(), weights_y.cpu(), M, .1), M):.5f}")
    #print(f"W2 = {ot.emd2(weights.cpu(), weights.cpu(), M).sqrt():.5f}")
    #del M

print(torch.cuda.memory_summary())

#sumd = [-slice_sum_keops(y, x, weights, scale, xi0,smooth_f_keops) / factor_C, -0.5* slice_sum_keops(x, x, weights, scale, xi0,smooth_f_keops) / factor_C] 
#sumr = [-slice_sum_keops(y, x, weights, scale, xi0,smooth_rest_f_keops)/factor_C, -0.5* slice_sum_keops(x, x, weights, scale, xi0,smooth_rest_f_keops)/factor_C] 
#sum1 = [torch.sum(fastsum(y, x, weights, scale, xi0)), 0.5* torch.sum(fastsum(x, x, weights, scale, xi0))]
#print(f'Riesz: {sum1[0]:.4f}  {sum1[1]:.4f}')
#print(f'smoothabs: {sumd[0]:.4f}  {sumd[1]:.4f}  direct')
