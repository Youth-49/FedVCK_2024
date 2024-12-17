import copy
import random

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
from torch.distributions.multivariate_normal import MultivariateNormal
from src.models import ResNet18, ConvNet, ResNet18BN
from .models import Projector
import torch.nn.functional as F

import torch
import torch.nn as nn
import logging

def get_gpu_mem_info(gpu_id=0):
    import pynvml
    pynvml.nvmlInit()
    gpu_id = int(str(gpu_id)[-1])
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(f'gpu_id {gpu_id} does not exsit!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    print(f"total {total}MB, used {used}MB, free {free}MB")
    return total, used, free

class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, device, temperature=0.07, z_dim=10, relation_class=None):
        super(SupervisedContrastiveLoss, self).__init__()
        self.device = device
        self.head = Projector(input_dim=z_dim, output_dim=z_dim).to(self.device)
        self.head.train()
        self.temperature = temperature
        self.relation_class = relation_class
        self.num_classes = num_classes

    def forward(self, x, y, proto=None, asymmetric=False):
        if asymmetric:
            x = self.head(x)
        x = F.normalize(x, dim=1)
        if proto is not None:
            sim_matrix = torch.exp(torch.matmul(x, proto.t()) / self.temperature)
        else:
            sim_matrix = torch.exp(torch.matmul(x, x.t()) / self.temperature)
        # generate the mask for positive and negative pairs
        mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
        relation_mask = torch.zeros_like(mask)
        for i in range(self.num_classes):
            class_mask = (y == i).unsqueeze(1).float()
            for cls in self.relation_class[i]:
                relation_mask += class_mask * (y == cls).unsqueeze(0).float()
        loss = -torch.log((mask * sim_matrix).sum(1) / (relation_mask * sim_matrix).sum(1)).mean()
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, z_dim, device, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.head = Projector(input_dim=z_dim, output_dim=z_dim).to(self.device)
        self.head.train()
        self.temperature = temperature

    def forward(self, x, proto, y, asymmetric=False):
        if asymmetric:
            x = self.head(x)
        x = F.normalize(x, dim=1)
        sim_matrix = torch.matmul(x, proto.t()) / self.temperature
        mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).float()
        mask = mask / mask.sum(dim=1, keepdim=True)
        loss = -(torch.log_softmax(sim_matrix, dim=1) * mask).sum(dim=1).mean()
        return loss

class RBF(nn.Module):
    def __init__(self, device='cpu', n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class PoliKernel(nn.Module):
    def __init__(self, constant_term=1, degree=2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree

    def forward(self, X):
        K = (torch.matmul(X, X.t()) + self.constant_term) ** self.degree
        return K

class LinearKernel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        K = torch.matmul(X, X.t())
        return K

class LaplaceKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] * (self.gammas)[:, None, None]).sum(dim=0)

class M3DLoss(nn.Module):
    def __init__(self, kernel_type, device):
        super().__init__()
        self.device = device
        if kernel_type == 'gaussian':
            self.kernel = RBF(device = self.device)
        elif kernel_type == 'linear':
            self.kernel = LinearKernel()
        elif kernel_type == 'polinominal':
            self.kernel = PoliKernel()
        elif kernel_type == 'laplace':
            self.kernel = LaplaceKernel()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class MMDLoss(nn.Module):
    '''
    https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据(n * len(x))
    target: 目标域数据(m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
        

def get_model(model_name, dataset_info):
    if model_name == "ConvNet":
        model = ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='instancenorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size']
        )
    elif model_name == "ConvNetBN":
        model = ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='batchnorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size']
        )
    elif model_name == "ResNet":
        model = ResNet18(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes']
        )
    elif model_name == 'ResNet18BN':
        model = ResNet18BN(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes']
        )
    else:
        raise NotImplementedError("only support ConvNet and ResNet")

    return model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def sample_random_model(model, rho):
    new_model = copy.deepcopy(model)
    parameters = new_model.parameters()

    mean = parameters.view(-1)
    multivariate_normal = MultivariateNormal(mean, torch.eye(mean.shape[0]))
    distance = rho + 1
    while distance > rho:
        sample = multivariate_normal.sample()
        distance = torch.sqrt(torch.sum((mean - sample)**2))

    new_parameters = sample.view(parameters.shape)
    for old_param, new_param in zip(parameters, new_parameters):
        with torch.no_grad():
            old_param.fill_(new_param)

    return new_model

def random_pertube(model, rho):
    new_model = copy.deepcopy(model)
    for p in new_model.parameters():
        gauss = torch.normal(mean=torch.zeros_like(p), std=1)
        if p.grad is None:
            p.grad = gauss
        else:
            p.grad.data.copy_(gauss.data)

    norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in new_model.parameters() if p.grad is not None]), p=2)

    with torch.no_grad():
        scale = rho / (norm + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        for p in new_model.parameters():
            if p.grad is not None:
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    new_model.zero_grad()
    return new_model


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.
    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


def get_daparam(dataset, model, model_eval):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN', 'ConvNet']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}