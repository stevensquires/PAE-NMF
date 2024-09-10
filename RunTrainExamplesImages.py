import PAE_NMF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def pathData():
    return 'Data/'
dataNames=['Vfaces.csv','Vfinance.csv','Vgenes.txt']
r=80
maxEpochs=3000
V=pd.read_csv(pathData()+dataNames[0],index_col=0,header=None).values
learning_rate=5e-2
batchSize=500#V.shape[1]
IDs=np.linspace(0,V.shape[1]-1,num=V.shape[1]).astype(int).tolist()
reduceLR=[1000,1400,1800,2200,2600]
errorStore,wkNumpy,wlNumpy,wfNumpy,V=PAE_NMF.runPAE_NMF(learning_rate,V,r,maxEpochs,batchSize,IDs,'cpu',reduceLR)
Vp,k,l=PAE_NMF.modelPredNumpy(wkNumpy,wlNumpy,wfNumpy,V)
figsize=4
plt.close('all')
randPerm=np.random.permutation(Vp.shape[1])
for i in range(5):
    plt.figure(i+1,figsize=(2*figsize,figsize))
    plt.subplot(1,2,1)
    im0=V[:,randPerm[i]].reshape(19,19)
    plt.imshow(im0.T,cmap='gray',vmin=0,vmax=1)
    plt.subplot(1,2,2)
    im1=Vp[:,randPerm[i]].reshape(19,19)
    plt.imshow(im1.T,cmap='gray',vmin=0,vmax=1)
    
plt.figure(6)
plt.plot(np.array(errorStore),'k--')
plt.ylim(0,60000)