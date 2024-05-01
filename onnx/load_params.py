import torch 
from torch_model import get_model

def load_params(model, ckpt_file):
    model_params = model.state_dict()
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
    state_dicts = checkpoint['state_dict']

    new_params = {}
    for key, value in zip(model_params.keys(), state_dicts.values()): 
        new_params[key] = value

    model_params.update(new_params)
    model.load_state_dict(model_params)

    print("Load model weights success")
    return model  