import numpy as np
import torch
import dataGenerator


def initialise(Vorig):
    V=Vorig/np.amax(Vorig.flatten())
    return V

def returnObjFn(yOut,y_pred,k,l):
    mse=(y_pred.T - yOut).pow(2).sum()
    D0=torch.log(k/(l.pow(k)))
    D1=(k-1)*(torch.log(l)-0.5772/k)
    D2=l*torch.exp(torch.lgamma((1/k)+1))-1
    Dkl1=D0+D1+D2
    Dkl=Dkl1.sum()
    loss=mse+Dkl
    return loss
def modelPred(wk,wl,wf,xIn,minK,minL,epsilon):
    k=(torch.matmul(xIn,wk.T)).clamp(min=minK) 
    l=(torch.matmul(xIn,wl.T)).clamp(min=minL) 
    h=l*torch.pow((-torch.log(epsilon)),(1/k))
    y_pred=torch.matmul(wf,h.T)
    return y_pred,k,l
def zeroGrads(wk,wl,wf):
    wk.grad.data.zero_()
    wl.grad.data.zero_()
    wf.grad.data.zero_()
def initialiseModel(m,r,meanV):
    low,high=0.3,0.7
    wk=torch.tensor((1/(meanV*m))*np.random.uniform(low=low, high=high,size=(r,m)), requires_grad=True).float()
    wk.retain_grad()
    wl=torch.tensor((1/(meanV*m))*np.random.uniform(low=low, high=high,size=(r,m)), requires_grad=True).float()
    wl.retain_grad()
    wf=torch.tensor((1/(meanV*m))*np.random.uniform(low=low, high=high,size=(m,r)), requires_grad=True).float()
    wf.retain_grad()
    return wk,wl,wf
def returnGrad(wToUpdate,learning_rate,name):
    update=learning_rate*wToUpdate.grad.data/(1e-9+torch.norm(wToUpdate.grad.data))
    if update.isfinite().all():
        wToUpdate.data-=update
    else:
        wToUpdate=wToUpdate
        print(name,'Not updated')
    return wToUpdate
def updateParams(wk,wl,wf,learning_rate): 
    wk=returnGrad(wk,learning_rate,'wk')
    wl=returnGrad(wl,learning_rate,'wl')
    wf=returnGrad(wf,learning_rate,'wf')
    wf.data=wf.data.clamp(min=0)
    return wk,wl,wf

def runPAE_NMF(learning_rate,Vorig,r,maxEpochs,batchSize,IDs,gpuOrCpu,reduceLR=[],minRand=0.35,maxRand=0.65):
    m=Vorig.shape[0]
    V=initialise(Vorig)
    
    trainingGenerator=dataGenerator.makeGenerators(V,IDs,minRand,maxRand,r,batchSize)
    if gpuOrCpu=='gpu':
        device=torch.device("cuda:0")
    elif gpuOrCpu=='cpu':
        device = torch.device("cpu") 
         
    minK,minL=0.001,0.001
    wk,wl,wf=initialiseModel(m,r,np.mean(V.flatten()))
    errorStore=[]
    minLoss=1e15
    for epoch in range(maxEpochs):
        with torch.set_grad_enabled(True):
            epochLoss=0
            for local_batch,epsilon,ID in trainingGenerator:
                
                local_batch = local_batch.float().to(device)
                localPreds,k,l=modelPred(wk,wl,wf,local_batch,minK,minL,epsilon)
                loss=returnObjFn(local_batch,localPreds,k,l)
                loss.backward()
                wk,wl,wf=updateParams(wk,wl,wf,learning_rate)
                zeroGrads(wk,wl,wf)
                epochLoss=epochLoss+loss.detach().cpu().numpy()
        if epoch in reduceLR:
            learning_rate=0.5*learning_rate
            print('Reducing learning rate')
        print(epoch,epochLoss)
        errorStore.append(epochLoss)
        if epochLoss<minLoss and ~np.isnan(epochLoss):
            wkOpt=wk.clone().detach()
            wlOpt=wl.clone().detach()
            wfOpt=wf.clone().detach()
        
    wkNumpy=wkOpt.detach().cpu().numpy()
    wlNumpy=wlOpt.detach().cpu().numpy()
    wfNumpy=wfOpt.detach().cpu().numpy()
    return errorStore,wkNumpy,wlNumpy,wfNumpy,V
def modelPredNumpy(wk,wl,wf,xIn,minK=0.000001,minL=0.000001,epsilon=0.5):
    k=np.clip(np.matmul(wk,xIn),a_min=minK,a_max=None)
    l=np.clip(np.matmul(wl,xIn,),a_min=minL,a_max=None)
    h=l*np.power((-np.log(epsilon)),(1/k))
    y_pred=np.matmul(wf,h)
    return y_pred,k,l








