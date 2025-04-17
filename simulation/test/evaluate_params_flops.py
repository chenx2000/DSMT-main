import option as opt
import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

from utils import my_summary
from model import model_generator

my_summary(model_generator(opt.method, opt.pretrained_model_path), 256, 256, 28, 1)
