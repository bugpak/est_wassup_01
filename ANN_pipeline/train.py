import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional
import numpy as np
import pandas as pd
from nn.preprocess import preprosess_Module
from nn.model import ANN
from nn.utils import CustomDataset
from tqdm.auto import tqdm
import argparse
from nn.validation import *
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset 

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#print(device)

def train(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(y)
  return total_loss/len(data_loader.dataset)

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
  '''
  model.eval()
  total_loss,correct = 0.,0.
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)
      #correct = (output > 0.5).astype(np.float32)
      #correct += (output.argmax(1) == y).type(torch.float).sum().item()
      if metric is not None:
        #print(output.squeeze(1).squeeze(),y.squeeze(1).squeeze())
        metric.update_state(output, y)
  #acc = correct / len(data_loader.dataset)
  total_loss = total_loss/len(data_loader.dataset)
  return total_loss 


def main(args):
  device = torch.device(args.device)

  submission_df = pd.read_csv(args.data_submission)
  train_df = pd.read_csv(args.data_train)
  test_df = pd.read_csv(args.data_test)
  preprocess = preprosess_Module(train_df)
  preprocess_ = preprosess_Module(test_df)
  X_trn, X_val = preprocess.get_X(train_df,test_df)
  y_trn = preprocess_.get_y(train_df)[:,np.newaxis]
  ds = CustomDataset(X_trn.astype(np.float32), y_trn.astype(np.float32))
  ds_val = CustomDataset(X_val.astype(np.float32))
  dl = DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle)
  dl_val = DataLoader(ds_val, batch_size=args.batch_size)

  model = ANN(X_trn.shape[-1] ,args.hidden_dim).to(device)
  print(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  
  if args.train is True:
    pbar = range(args.epochs)
    if args.pbar:
      pbar = tqdm(pbar)
    
    print("Learning Start!")
    for _ in pbar:
      loss = train(model, nn.functional.mse_loss, optimizer, dl, device)
      pbar.set_postfix(trn_loss=loss)
    #evaluate(model, nn.functional.binary_cross_entropy, dl, device)    
    print("Done!")
    torch.save(model.state_dict(), args.output)
    
    model = ANN(X_trn.shape[-1] ,args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.output))
    model.eval()
    
    pred = []
    with torch.inference_mode():
      for x in dl_val:
        x = x[0].to(device)
        out = model(x)
        out = torch.round(out)
        pred.append(out.detach().cpu().numpy())
    
    submission_df['ECLO'] = [int(i) for i in np.concatenate(pred).squeeze()]
    submission_df.to_csv(args.submission,index=False)
  
  print('------------------------------------------------------------------')
  if args.validation == True:
    scores = Validation(X_trn, y_trn)
    scores = pd.DataFrame(scores.kfold(model, n_splits=5, epochs=args.epochs, shuffle=True, random_state=2023))
    print(pd.concat([scores, scores.apply(['mean', 'std'])]))
    
  return


def get_args_parser(add_help=True):

  parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

  parser.add_argument("--data-submission", default="/home/estsoft/data/sample_submission.csv", type=str, help="submission dataset path")
  parser.add_argument("--data-train", default="/home/estsoft/data/train.csv", type=str, help="train dataset path")
  parser.add_argument("--data-test", default="/home/estsoft/data/test.csv", type=str, help="test dataset path")
  parser.add_argument("--hidden-dim", default=64, type=int, help="dimension of hidden layer")
  parser.add_argument("--device", default="cuda", type=str, help="device (Use cpu/cuda/mps)")
  parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
  parser.add_argument("--shuffle", default=True, type=bool, help="shuffle")
  parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
  parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
  parser.add_argument("--pbar", default=True, type=bool, help="progress bar")
  parser.add_argument("-o", "--output", default="./model.pth", type=str, help="path to save output model")
  parser.add_argument("-sub", "--submission", default="./submission.csv", type=str, help="path to save submission")
  parser.add_argument("-train", "--train", default=False, type=bool, help="full data set train")
  parser.add_argument("-val", "--validation", default=False, type=bool, help="kfold cross validation train")
  
  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  main(args)