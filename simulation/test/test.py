from option import opt
import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

from model import *
from utils import *
import scipy.io as scio
import torch
import numpy as np

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)


def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()
    with torch.no_grad():
        model_out = model(input_meas, input_mask)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()
    return pred, truth


def main():
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    load_checkpoint(opt.pretrained_model_path, model)
    pred, truth = test(model)
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})


if __name__ == '__main__':
    main()