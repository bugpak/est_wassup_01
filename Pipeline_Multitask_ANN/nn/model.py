import torch.nn as nn

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
        nn.Linear(input,hidden),   
        nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden,hidden*2),  
        nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden*2,hidden*4),
        nn.ReLU(),
        #nn.Dropout(0.3),    
        nn.Linear(hidden*4,hidden*4),
        nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden*4,4),        
        nn.ReLU()
        )
    
  def forward(self, x:list):
    x = self.linear_stack(x)
    return x