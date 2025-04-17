import torch
from .DSMT import DSMT


def model_generator(method, pretrained_model_path=None):
    if method == 'dsmt':
        model = DSMT(num_blocks=[1, 4, 6]).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
    #     checkpoint = torch.load(pretrained_model_path)
    #     model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=True)
    return model
