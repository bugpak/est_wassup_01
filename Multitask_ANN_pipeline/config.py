import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.regression import MeanSquaredLogError
# from nn import ANN, RMSLELoss

config = {
    "files": {
        "data_submission": "/home/estsoft/data/sample_submission.csv",
        "data_train": "/home/estsoft/data/train.csv",
        "data_test":"/home/estsoft/data/test.csv",
        "output": "./submit/model_",
        "submission":"./submit/submission_",
        "name": ""
    },
    "model_params": {
        "hidden_dim": 64,
        "use_dropout": True,
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 64,
            "shuffle": True,
        },
        "optim_params": {"lr": 0.001, },
        "device": "cuda",
        "epochs": 5,
        "pbar": True,
        "min_delta": 0,
        "patience": 5,
    },
    "train": True,
    "validation": True,
}
