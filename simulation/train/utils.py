import torch
import numpy as np
import scipy.io as sio
import os
import random
from ssim_torch import ssim
import logging
import re
from fvcore.nn import FlopCountAnalysis

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    return Phi_batch, Phi_s_batch


def init_mask(mask_path, mask_type, batch_size):
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == 'Phi':  # Use the shifted mask
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':  # Use the original mask
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask


def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    print('training sences:', len(scene_list))
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num <= 205:
            if 'mat' not in scene_path:
                continue
            data = sio.loadmat(scene_path)
            if "img_expand" in data:
                img = data['img_expand'] / 65536.
            elif "img" in data:
                img = data['img'] / 65536.
            elif "data_slice" in data:
                img = data['data_slice'] / 65536.
            img = img.astype(np.float32)
            imgs.append(img)
            print('Sence {} is loaded. {}'.format(i + 1, scene_list[i]))
    return imgs


def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1], 256, 256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c, h, w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def shuffle_crop(train_data, batch_size, crop_size=256, argument=True):
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size // 2)
        processed_data = np.zeros((batch_size // 2, crop_size, crop_size, 28), dtype=np.float32)
        h, w, _ = train_data[0].shape
        for i in range(batch_size // 2):
            img = train_data[index[i]]
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        processed_data = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h - crop_size // 2)
                y_index = np.random.randint(0, w - crop_size // 2)
                processed_data[j] = train_data[sample_list[j]][x_index:x_index + crop_size // 2,
                                    y_index:y_index + crop_size // 2, :]
            gt_batch_2 = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).cuda()
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size,
                                         y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch


def shift_back(inputs, step=2):
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    # Simulate the process of imaging system
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas


def init_meas(gt, mask, input_setting):
    if input_setting == 'H':  # Shift the data back before inputting it into the network
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':  # Shift the data back and modulate it by mask before inputting it into the network
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':  # Input the data into network directly
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas


def torch_psnr(img, ref):  # input [28,256,256]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def checkpoint(model, epoch, optimizer, scheduler, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


# Load checkpoint
def load_checkpoint(filepath, model, optimizer, scheduler):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
