import torch.optim as optim
from torch.optim import AdamW
import torch.nn as nn

def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("train", "optimizer")
    
    learning_rate = config.getfloat("train", "learning_rate")
    
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate,
                          weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError
    # optimizer=nn.DataParallel(optimizer)
    return optimizer
