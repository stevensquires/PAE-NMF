import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.append('/home/steve/Documents/Research/NMF-autoencoders/Modules')
import torch
#import getData
from torch.autograd import Variable


def runPAE_NMF(learning_rate,V,r,paramInit,epochNum,batchNum):
    n=V.shape[1]
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dtype = torch.cuda.FloatTensor

    x = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    y = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    wk = Variable(torch.from_numpy(paramInit[0]).type(dtype),requires_grad=True)
    wl = Variable(torch.from_numpy(paramInit[1]).type(dtype),requires_grad=True)
    wFinal = Variable(torch.from_numpy(paramInit[4]).type(dtype),requires_grad=True)    
    
    iterNum=int(epochNum*(n/batchNum))
    DklStore=torch.zeros(iterNum,1)
    errorsBatch=torch.zeros(iterNum,1)
    errorsEpoch=np.zeros((epochNum,1),float)
#    minRand,maxRand=0.001,0.999
    minK,minL=0.000001,0.000001
    iterVal=0
    for i in range(epochNum):
        randPerm=np.random.permutation(n)
        for j in range(int(n/batchNum)):
            xIn=x[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            yOut=y[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            k=(torch.matmul(wk,xIn)).clamp(min=minK) # +biasA
            l=(wl.matmul(xIn)).clamp(min=minL) # +biasB
           
#            epsilon=(maxRand-minRand)*torch.rand(r,batchNum)+minRand
            epsilon=0.5*torch.ones(r,batchNum)
            h=l*torch.pow((-torch.log(epsilon)),(1/k))
            y_pred=wFinal.matmul(h)
            mse=(y_pred - yOut).pow(2).sum()
            Dkl1=torch.log(k/(l.pow(k)))+(k-1)*(torch.log(l)-0.5772/k)+l*torch.exp(torch.lgamma((1/k)+1))-1
            Dkl=Dkl1.sum()
            loss =mse + Dkl
            loss.backward()
            
            wk.data -= learning_rate * wk.grad.data#/(1e-9+torch.norm(wk.grad.data))
            wl.data -= learning_rate * wl.grad.data#/(1e-9+torch.norm(wl.grad.data))
            
            wFinal.data -= learning_rate * wFinal.grad.data#/(1e-9+torch.norm(wFinal.grad.data))
            wFinal.data = wFinal.data.clamp(min=0)
            
            wk.grad.data.zero_()
            wl.grad.data.zero_()
            wFinal.grad.data.zero_()
            
            errorsBatch[iterVal,0]=loss.item()
            DklStore[iterVal,0]=Dkl.item()
            print(iterVal,loss.item())
            iterVal=iterVal+1
        K=wk.matmul(x).clamp(min=minK).cpu().detach().numpy()
        L=wl.matmul(x).clamp(min=minL).cpu().detach().numpy()
        H=L*np.power(-np.log(0.5),(1/K))
        Wfinal=wFinal.cpu().data.numpy()
        Vp=np.matmul(Wfinal,H)
        E=Vp-V
        errorsEpoch[i,0]=np.power(E,2).sum()
#        print(i+1,errorsEpoch[i,0])
    paramsFinal=[wk.cpu().data.numpy(),wl.cpu().data.numpy(),Wfinal,K,L,H]
    return paramsFinal,errorsBatch.cpu().data.numpy(),errorsEpoch,DklStore.cpu().data.numpy()

def runPAE_NMF_noKL(learning_rate,V,r,paramInit,epochNum,batchNum):
    n=V.shape[1]
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dtype = torch.cuda.FloatTensor

    x = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    y = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    wk = Variable(torch.from_numpy(paramInit[0]).type(dtype),requires_grad=True)
    wl = Variable(torch.from_numpy(paramInit[1]).type(dtype),requires_grad=True)
    wFinal = Variable(torch.from_numpy(paramInit[4]).type(dtype),requires_grad=True)    
    
    iterNum=int(epochNum*(n/batchNum))
#    DklStore=torch.zeros(iterNum,1)
    errorsBatch=torch.zeros(iterNum,1)
    errorsEpoch=np.zeros((epochNum,1),float)
    #    minRand,maxRand=0.001,0.999
    minK,minL=0.000001,0.000001
    iterVal=0
    for i in range(epochNum):
        randPerm=np.random.permutation(n)
        for j in range(int(n/batchNum)):
            xIn=x[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            yOut=y[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            k=(torch.matmul(wk,xIn)).clamp(min=minK) # +biasA
            l=(wl.matmul(xIn)).clamp(min=minL) # +biasB
           
#            epsilon=(maxRand-minRand)*torch.rand(r,batchNum)+minRand
            epsilon=0.5*torch.ones(r,batchNum)
            h=l*torch.pow((-torch.log(epsilon)),(1/k))
            y_pred=wFinal.matmul(h)
            mse=(y_pred - yOut).pow(2).sum()
#            Dkl1=torch.log(k/(l.pow(k)))+(k-1)*(torch.log(l)-0.5772/k)+l*torch.exp(torch.lgamma((1/k)+1))-1
#            Dkl=Dkl1.sum()
            loss = mse # + Dkl
            loss.backward()
            
            wk.data -= learning_rate * wk.grad.data#/(1e-9+torch.norm(wk.grad.data))
            wl.data -= learning_rate * wl.grad.data#/(1e-9+torch.norm(wl.grad.data))
            
            wFinal.data -= learning_rate * wFinal.grad.data#/(1e-9+torch.norm(wFinal.grad.data))
            wFinal.data = wFinal.data.clamp(min=0)
            
            wk.grad.data.zero_()
            wl.grad.data.zero_()
            wFinal.grad.data.zero_()
            
            errorsBatch[iterVal,0]=loss.item()
#            DklStore[iterVal,0]=Dkl.item()
            iterVal=iterVal+1
            print(iterVal,loss.item())
        K=wk.matmul(x).clamp(min=0).cpu().detach().numpy()
        L=wl.matmul(x).clamp(min=0).cpu().detach().numpy()
        H=L*np.power(-np.log(0.5),(1/K))
        Wfinal=wFinal.cpu().data.numpy()
        Vp=np.matmul(Wfinal,H)
        E=Vp-V
        errorsEpoch[i,0]=np.power(E,2).sum()
#        print(i+1,errorsEpoch[i,0])
    paramsFinal=[wk.cpu().data.numpy(),wl.cpu().data.numpy(),Wfinal,K,L,H]
    return paramsFinal,errorsBatch.cpu().data.numpy(),errorsEpoch

def runPAE_NMF2(learning_rate,V,r,paramInit,epochNum,batchNum):
    n=V.shape[1]
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dtype = torch.cuda.FloatTensor

    x = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    y = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    wk = Variable(torch.from_numpy(paramInit[0]).type(dtype),requires_grad=True)
    wl = Variable(torch.from_numpy(paramInit[1]).type(dtype),requires_grad=True)
    wFinal = Variable(torch.from_numpy(paramInit[4]).type(dtype),requires_grad=True)    
      
    iterNum=int(epochNum*(n/batchNum))
    DklStore=torch.zeros(iterNum,1)
    errorsBatch=torch.zeros(iterNum,1)
    errorsEpoch=np.zeros((epochNum,1),float)
    minRand,maxRand=0.001,0.999
    minK,minL=0.000001,0.000001
    iterVal=0
    for i in range(epochNum):
        randPerm=np.random.permutation(n)
        for j in range(int(n/batchNum)):
            xIn=x[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            yOut=y[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            k=(torch.matmul(wk,xIn)).clamp(min=minK) # +biasA
            l=(wl.matmul(xIn)).clamp(min=minL) # +biasB
           
            epsilon=(maxRand-minRand)*torch.rand(r,batchNum)+minRand
            h=l*torch.pow((-torch.log(epsilon)),(1/k))
            y_pred=wFinal.matmul(h)
            mse=(y_pred - yOut).pow(2).sum()
            Dkl1=torch.log(k/(l.pow(k)))+(k-1)*(torch.log(l)-0.5772/k)+l*torch.exp(torch.lgamma((1/k)+1))-1
            Dkl=Dkl1.sum()
            loss =mse + Dkl
            loss.backward()
            
            wk.data -= learning_rate * wk.grad.data#/(1e-9+torch.norm(wk.grad.data))
            wl.data -= learning_rate * wl.grad.data#/(1e-9+torch.norm(wl.grad.data))
            
            wFinal.data -= learning_rate * wFinal.grad.data#/(1e-9+torch.norm(wFinal.grad.data))
            wFinal.data = wFinal.data.clamp(min=0)
            
            wk.grad.data.zero_()
            wl.grad.data.zero_()
            wFinal.grad.data.zero_()
            
            errorsBatch[iterVal,0]=loss.item()
            DklStore[iterVal,0]=Dkl.item()
            iterVal=iterVal+1
            print(iterVal,loss.item())
        K=wk.matmul(x).clamp(min=0).cpu().detach().numpy()
        L=wl.matmul(x).clamp(min=0).cpu().detach().numpy()
        H=L*np.power(-np.log(0.5),(1/K))
        Wfinal=wFinal.cpu().data.numpy()
        Vp=np.matmul(Wfinal,H)
        E=Vp-V
        errorsEpoch[i,0]=np.power(E,2).sum()
#        print(i+1,errorsEpoch[i,0])
    K=wk.matmul(x).clamp(min=0).cpu().detach().numpy()
    L=wl.matmul(x).clamp(min=0).cpu().detach().numpy()
    H=L*np.power(-np.log(0.5),(1/K))
    Wfinal=wFinal.cpu().data.numpy()
    paramsFinal=[wk.cpu().data.numpy(),wl.cpu().data.numpy(),Wfinal,K,L,H]
    return paramsFinal,errorsBatch.cpu().data.numpy(),errorsEpoch,DklStore.cpu().data.numpy()

def runPAE_NMF_noKL2(learning_rate,V,r,paramInit,epochNum,batchNum):
    n=V.shape[1]
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    dtype = torch.cuda.FloatTensor

    x = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    y = Variable(torch.from_numpy(V).type(dtype),requires_grad=False)
    wk = Variable(torch.from_numpy(paramInit[0]).type(dtype),requires_grad=True)
    wl = Variable(torch.from_numpy(paramInit[1]).type(dtype),requires_grad=True)
    wFinal = Variable(torch.from_numpy(paramInit[4]).type(dtype),requires_grad=True)    
    
    iterNum=int(epochNum*(n/batchNum))
#    DklStore=torch.zeros(iterNum,1)
    errorsBatch=torch.zeros(iterNum,1)
    errorsEpoch=np.zeros((epochNum,1),float)
    minRand,maxRand=0.001,0.999
    minK,minL=0.000001,0.000001
    iterVal=0
    for i in range(epochNum):
        randPerm=np.random.permutation(n)
        for j in range(int(n/batchNum)):
            xIn=x[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            yOut=y[:,randPerm[j*batchNum:j*batchNum+batchNum]]
            k=(torch.matmul(wk,xIn)).clamp(min=minK) # +biasA
            l=(wl.matmul(xIn)).clamp(min=minL) # +biasB         
            epsilon=(maxRand-minRand)*torch.rand(r,batchNum)+minRand
            h=l*torch.pow((-torch.log(epsilon)),(1/k))
            y_pred=wFinal.matmul(h)
            mse=(y_pred - yOut).pow(2).sum()
#            Dkl1=torch.log(k/(l.pow(k)))+(k-1)*(torch.log(l)-0.5772/k)+l*torch.exp(torch.lgamma((1/k)+1))-1
#            Dkl=Dkl1.sum()
            loss =mse # + Dkl
            loss.backward()
            
            wk.data -= learning_rate * wk.grad.data#/(1e-9+torch.norm(wk.grad.data))
            wl.data -= learning_rate * wl.grad.data#/(1e-9+torch.norm(wl.grad.data))
            
            wFinal.data -= learning_rate * wFinal.grad.data#/(1e-9+torch.norm(wFinal.grad.data))
            wFinal.data = wFinal.data.clamp(min=0)
            
            wk.grad.data.zero_()
            wl.grad.data.zero_()
            wFinal.grad.data.zero_()
            
            errorsBatch[iterVal,0]=loss.item()
            print(iterVal,loss.item())
#            DklStore[iterVal,0]=Dkl.item()
            iterVal=iterVal+1
        K=wk.matmul(x).clamp(min=0).cpu().detach().numpy()
        L=wl.matmul(x).clamp(min=0).cpu().detach().numpy()
        H=L*np.power(-np.log(0.5),(1/K))
        Wfinal=wFinal.cpu().data.numpy()
        Vp=np.matmul(Wfinal,H)
        E=Vp-V
        errorsEpoch[i,0]=np.power(E,2).sum()
#        print(i+1,errorsEpoch[i,0])
    paramsFinal=[wk.cpu().data.numpy(),wl.cpu().data.numpy(),Wfinal,K,L,H]
    return paramsFinal,errorsBatch.cpu().data.numpy(),errorsEpoch














