
config = {
    "files": {
        "data_submission": "/home/estsoft/data/sample_submission.csv",
        "data_train": "/home/estsoft/data/train.csv",
        "data_test":"/home/estsoft/data/test.csv",
        "output": "./submit/model_",
        "submission":"./submit/submission_",
        "name": "3hidden_dim32_notdrop_nonexternal_0.001_rsme_epoch200_"
    },
    "model_params": {
        "hidden_dim": 32,
        "use_dropout": True,
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 64,
            "shuffle": True,
        },
        "optim_params": {"lr": 0.001, },
        "device": "cuda",
        "epochs": 200,
        "pbar": True,
        "min_delta": 0,
        "patience": 100,
    },
    "train": True,
    "validation": True,
}
