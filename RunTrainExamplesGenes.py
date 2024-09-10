import PAE_NMF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pathData():
    return 'Data/'
dataNames=['Vfaces.csv','Vfinance.csv','Vgenes.txt']
V0=pd.read_csv(pathData()+dataNames[2],index_col=0,header=None,sep='\t').values
V=np.zeros((V0.shape),float)
for i in range(V0.shape[1]):
    V[:,i]=V0[:,i]/np.amax(V0[:,i])
r=4
maxEpochs=20000
learning_rate=5e-4
batchSize=V.shape[1]
IDs=np.linspace(0,V.shape[1]-1,num=V.shape[1]).astype(int).tolist()
reduceLR=[]
errorStore,wkNumpy,wlNumpy,wfNumpy,V=PAE_NMF.runPAE_NMF(learning_rate,V,r,maxEpochs,batchSize,IDs,'cpu',reduceLR)
Vp,k,l=PAE_NMF.modelPredNumpy(wkNumpy,wlNumpy,wfNumpy,V)
figsize=4
plt.close('all')
plt.figure(1)
randPerm=np.random.permutation(V.shape[1])
for i in range(5):    
    plt.plot(V[:,randPerm[i]],Vp[:,randPerm[i]],'k.')
plt.axis([0,1,0,1])
plt.grid()
plt.xlabel('Real=');plt.ylabel('Reconstructed')
plt.figure(2)
plt.plot(np.array(errorStore),'k--')
