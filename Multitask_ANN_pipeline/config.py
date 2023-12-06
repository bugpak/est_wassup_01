import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.regression import MeanSquaredLogError
# from nn import ANN, RMSLELoss

config = {
    "files": {
        # "output": "./model_128_64_32_0.001_도시drop_epoch10_batch64.pth",
        # "output_csv": "./results/five_fold.csv",
        "data_submission": "/home/estsoft/data/sample_submission.csv",
        "data_train": "/home/estsoft/data/train.csv",
        "data_test":"/home/estsoft/data/test.csv",
        "output": "./submit/model_",
        "submission":"./submit/submission_",
        "name": ""
    },
    # "model": ANN,
    "model_params": {
        # "input_dim": "auto",  # Always will be determined by the data shape
        # "hidden_dim": [128, 64, 32],
        "hidden_dim": 64,
        "use_dropout": True,
        # "dropout_ratio": 0.3,
        # "activation": torch.nn.ReLU(),
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 64,
            "shuffle": True,
        },
        # "loss": F.mse_loss,
        # "optim": torch.optim.Adam,
        "optim_params": {"lr": 0.001, },
        "device": "cuda",
        # "metric": RMSLELoss(),
        "epochs": 50,
        "pbar": True,
        "min_delta": 0,
        "patience": 5,
    },
    # "cv_params": {"n_split": 5, },
    # "train": False,
    # "validation": False,
    "train": True,
    "validation": True,
}
