import torch
import torch.nn as nn

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        #print('1',x[:,0], y)
        loss = torch.sqrt(criterion(x[:,0], y[:,0]))*10+torch.sqrt(criterion(x[:,1], y[:,1]))*5+torch.sqrt(criterion(x[:,2], y[:,2]))*3+torch.sqrt(criterion(x[:,3], y[:,3]))
        
        return loss