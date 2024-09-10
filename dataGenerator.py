import numpy as np
from torch.utils import data
import torch
## this class returns a 3 channel image, normalised to be fed into the pytorch 
## models.

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self,V,IDs,minRand,maxRand,r): ### 
        'Initialization'
        self.V=V
        self.IDs=IDs
        self.minRand=minRand
        self.maxRand=maxRand
        self.r=r
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.IDs)
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample    
        ID = self.IDs[index]
        vIn= self.V[:,ID]
        epsilon=(self.maxRand-self.minRand)*torch.rand(self.r)+self.minRand
        return vIn,epsilon,ID
### call to make generators: partition is the training and testing split

        
def makeGenerators(V,IDs,minRand,maxRand,r,batchSize): ### 
    training_set=Dataset(V,IDs,minRand,maxRand,r)
    paramsTrain = {'batch_size':batchSize,'shuffle': False}
    training_generator = data.DataLoader(training_set, **paramsTrain)
    return training_generator











