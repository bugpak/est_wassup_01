import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
from nn.utils import CustomDataset

class ANN(nn.Module):
  '''
  function for building Neural network 
  
  this neural network is composed two hidden layer
  
  args:
    input: int
    hidden: int
    
  '''
  def __init__(self, input:int=5, hidden:int=64):
    super().__init__()
    self.linear_stack = nn.Sequential(
        nn.Linear(input,hidden),     #(13,64)       (13, 32)
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(hidden,hidden*2),  #(64,128)       (32, 64)
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(hidden*2,hidden*2), #(128,256)       (64, 128)
        #nn.ReLU(),
        #nn.Dropout(0.3),     
        # nn.Linear(hidden*4,hidden*2), #(256, 64)       (128, 64)
        #nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(hidden*2,4),          #(64,4)       (64, 4)
        nn.ReLU()
        )
    
  def forward(self, x:list):
    x = self.linear_stack(x)
    return x