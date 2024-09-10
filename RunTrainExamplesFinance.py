import PAE_NMF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def pathData():
    return 'Data/'
dataNames=['Vfaces.csv','Vfinance.csv','Vgenes.txt']
r=5
maxEpochs=20000
V0=pd.read_csv(pathData()+dataNames[1],index_col=0,header=None).values
V=np.zeros((V0.shape),float)
for i in range(V0.shape[1]):
    V[:,i]=V0[:,i]/np.amax(V0[:,i])


learning_rate=5e-2
batchSize=V.shape[1]
IDs=np.linspace(0,V.shape[1]-1,num=V.shape[1]).astype(int).tolist()
reduceLR=[1000,1400,1800,2200,2600]
errorStore,wkNumpy,wlNumpy,wfNumpy,V=PAE_NMF.runPAE_NMF(learning_rate,V,r,maxEpochs,batchSize,IDs,'cpu',reduceLR)
Vp,k,l=PAE_NMF.modelPredNumpy(wkNumpy,wlNumpy,wfNumpy,V)
figsize=4
plt.close('all')
randPerm=np.random.permutation(Vp.shape[1])
plt.figure(1,figsize=(5*figsize,figsize))
for i in range(10):
    plt.subplot(2,5,1+i)
    plt.plot(V[:,randPerm[i]],'b--')
    plt.plot(Vp[:,randPerm[i]],'r--')
    
    
plt.figure(6)
plt.plot(np.array(errorStore),'k--')
plt.ylim(0,60000)