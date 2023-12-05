import numpy as np
import pandas as pd
import torch
from nn.model import ANN
from nn.utils import CustomDataset
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.metrics import accuracy_score
import tensorflow as tf
from torch import nn
from tqdm.auto import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from train import train, evaluate 
from copy import deepcopy
from nn.early_stop import EarlyStopper
from nn.rmsle import RMSLELoss, RMSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings(action='ignore')

class Validation:
  def __init__(self, X_trn, y_trn, patience, delta):
    self.patience = patience
    self.delta = delta
    self.X, self.y = 0,0 
    self.X_trn = X_trn
    self.y_trn = y_trn
    self.pred = 0
    self.scores={
    'MSE':[],
    'RMSE':[],
    'MAE':[],
    'RMSLE':[],
    'ACCURACY':[]
    }
    return
  
  def kfold(self, model, n_splits, shuffle=True, lr=0.001, epochs=100, batch=64, random_state=2023, device='cpu'):
    print(batch)
    X_val, y_val = 0,0
    n_splits = n_splits
    print(lr)
    skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    nets = [deepcopy(model).to(device) for i in range(n_splits)]
    #history = []

    for i, (trn_idx, val_idx) in enumerate(skf.split(self.X_trn, self.y_trn)):
      self.X, self.y = torch.tensor(self.X_trn[trn_idx]), torch.tensor(self.y_trn[trn_idx])
      X_val, y_val = torch.tensor(self.X_trn[val_idx]), torch.tensor(self.y_trn[val_idx])

      ds = TensorDataset(self.X, self.y)
      ds_val = TensorDataset(X_val, y_val)
      # ds = CustomDataset(X, y)
      # ds_val = CustomDataset(X_val, y_val)
      dl = DataLoader(ds, batch, shuffle=True)
      dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)
      print(dl)
      net = nets[i]
      optimizer = torch.optim.AdamW(net.parameters(), lr)
      scheduler = ReduceLROnPlateau(optimizer,'min',factor=0.8,patience=3,min_lr=0.000001)
      
      pbar = range(epochs)
      pbar = tqdm(pbar)
      early_stopper = EarlyStopper(self.patience, self.delta)
      for j in pbar:
        accuracy = tf.keras.metrics.Accuracy()
        loss = train(net, RMSELoss(), optimizer, dl, device)
        loss_val = evaluate(net, RMSELoss(), dl_val, device, accuracy)
        scheduler.step(loss_val)
        acc_val = accuracy.result().numpy()
        pbar.set_postfix(trn_loss=loss, val_loss=loss_val, val_acc=acc_val)
        if early_stopper.early_stop(net,validation_loss=loss_val, mode=False):             
          break
      net = nets[i].to(device)
      self.pred = net(self.X)
      self.pred = torch.round(self.pred)
      self.loss_functoin()
      
    return self.scores
  
  def loss_functoin(self):
    self.y = [int(i) for i in self.y.detach().numpy()]
    self.pred = [int(i) for i in self.pred.detach().numpy()]
    MSE = mean_squared_error(self.y, self.pred)
    RMSE = mean_squared_error(self.y, self.pred, squared=False)
    MAE = mean_absolute_error(self.y, self.pred)
    RMSLE = mean_squared_log_error(self.y, self.pred, squared=False)
    ACCURACY = accuracy_score(self.y, self.pred)
    self.scores['MSE'].append(np.sqrt(MSE))
    self.scores['RMSE'].append(np.sqrt(RMSE))
    self.scores['MAE'].append(np.sqrt(MAE))
    self.scores['RMSLE'].append(np.sqrt(RMSLE))
    self.scores['ACCURACY'].append(ACCURACY)
    
    return 
  
  def __call__(self):
    
    return