
config = {
    "files": {
        "data_submission": "../data/sample_submission.csv",
        "data_train": "../data/train.csv",
        "data_test":"../data/test.csv",
        "output": "./submit/model_",
        "submission":"./submit/submission_",
        "name": "hidden_dim32_notdrop_nonexternal_0.001_rsme_epoch1000_early200"
    },
    "model_params": {
        "hidden_dim": 32,
        "use_dropout": True,
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 128,
            "shuffle": True,
        },
        "optim_params": {"lr": 0.001, },
        "device": "cuda",
        "epochs": 1000,
        "pbar": True,
        "min_delta": 0,
        "patience": 200,
    },
    "train": True,
    "validation": True,
}
