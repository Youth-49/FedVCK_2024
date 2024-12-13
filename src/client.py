import copy
import os
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from sklearn.cluster import KMeans
import time
import logging
import math
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset

from dataset.data.dataset import PerLabelDatasetNonIID
from src.utils import sample_random_model, random_pertube, DiffAugment, ParamDiffAug, get_model, MMDLoss, M3DLoss

def get_gpu_mem_info(gpu_id=0):
    import pynvml
    pynvml.nvmlInit()
    gpu_id = int(str(gpu_id)[-1])
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        logging.info(f'gpu_id {gpu_id} does not exsit!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    logging.info(f"total {total}MB, used {used}MB, free {free}MB")
    return total, used, free

class Client:
    def __init__(
        self,
        cid: int,
        # --- dataset information ---
        train_set: PerLabelDatasetNonIID,
        classes: list[int],
        dataset_info: dict,
        # --- data condensation params ---
        ipc: int,
        compression_ratio: float,
        dc_iterations: int,
        real_batch_size: int,
        image_lr: float,
        image_momentum: float,
        image_weight_decay: float,
        lr: float,
        momentum: float,
        weight_decay: float,
        local_ep: int,
        dsa: bool,
        dsa_strategy: str,
        init: str,
        clip_norm: float,
        gamma: float,
        lamda: float,
        b: float,
        con_temp: float,
        kernel: str,
        save_root_path: str,
        device: torch.device,
    ):
        self.cid = cid

        self.train_set = train_set
        self.classes = classes
        self.dataset_info = dataset_info

        self.ipc = ipc
        self.compression_ratio = compression_ratio
        self.dc_iterations = dc_iterations
        self.real_batch_size = real_batch_size
        self.image_lr = image_lr
        self.image_momentum = image_momentum
        self.image_weight_decay = image_weight_decay
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.round = -1
        self.local_ep = local_ep
        self.dsa = dsa
        self.dsa_strategy = dsa_strategy
        self.model_name = None
        self.global_model = None
        self.prev_global_model = None
        self.dsa_param = ParamDiffAug()
        self.init = init
        self.clip_norm = clip_norm
        self.gamma = gamma
        self.lamda = lamda
        self.b = b
        self.con_temp = con_temp
        self.kernel = kernel
        self.save_root_path = save_root_path
        self.device = device

        if len(self.classes) > 0:
            if self.compression_ratio > 0.:
                self.ipc_dict = {c: max(5, int(math.ceil(len(self.train_set.indices_class[c])*self.compression_ratio))) for c in self.classes}
            else:
                self.ipc_dict = {c: self.ipc for c in self.classes}
            num_synthetic_images = sum(self.ipc_dict.values())
            self.accumulate_num_syn_imgs = [0,]
            for i, c in enumerate(self.classes):
                self.accumulate_num_syn_imgs.append(self.accumulate_num_syn_imgs[-1] + self.ipc_dict[c])

            self.synthetic_images = torch.randn(
                size=(
                    num_synthetic_images,
                    dataset_info['channel'],
                    dataset_info['im_size'][0],
                    dataset_info['im_size'][1],
                ),
                dtype=torch.float,
                requires_grad=True,
                device=self.device,
            )
            self.synthetic_labels = torch.cat([torch.ones(self.ipc_dict[c]) * c for c in self.classes]).long().to(self.device)


    def train_weighted_sample(self):
        self.round += 1
        # initialize S_k and initialize optimizer
        self.initialization()
        logging.info("synthesize from random noise")
        optimizer_image = torch.optim.SGD([self.synthetic_images,], lr=self.image_lr, momentum=self.image_momentum, weight_decay=self.image_weight_decay)
        optimizer_image.zero_grad()
        logging.info(f"client {self.cid} have real samples {[len(self.train_set.indices_class[c]) for c in self.classes]}")
        logging.info(f"client {self.cid} will condense {self.ipc_dict} samples for each class it owns")
        
        if self.round == 0:
            self.global_model = get_model(self.model_name, self.dataset_info).to(self.device)
        prototypes = self.get_feature_prototype()
        logit_prototypes = self.get_logit_prototype()

        logging.info(f"loss weighted matching the samples")
        self.train_set.cal_loss(copy.deepcopy(self.global_model), copy.deepcopy(self.prev_global_model), lamda=self.lamda, gamma=self.gamma, b=self.b, rounds=self.round, cid=self.cid, save_root_path=self.save_root_path)
        self.train_set.pre_sample(it=self.dc_iterations+1, bs=self.real_batch_size)
        
        total_loss = 0.
        self.global_model.train()
        for param in list(self.global_model.parameters()):
            param.requires_grad = False
        
        for dc_iteration in range(self.dc_iterations+1):
            loss = torch.tensor(0.0).to(self.device)
            images_real_all = []
            images_syn_all = []
            num_real_image = [0, ]
            for i, c in enumerate(self.classes):
                real_image = self.train_set.images_all[self.train_set.sample_indices[c][dc_iteration:dc_iteration+self.real_batch_size]]
                # real_image = self.train_set.weighted_sample(c, dc_iteration, self.real_batch_size)
                num_real_image.append(num_real_image[-1] + real_image.shape[0]) 
                synthetic_image = self.synthetic_images[self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i+1]].reshape(
                    (self.ipc_dict[c], self.dataset_info['channel'], self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))

                if self.dsa:
                    seed = int(time.time() * 1000) % 100000
                    real_image = DiffAugment(real_image, self.dsa_strategy, seed=seed, param=self.dsa_param)
                    synthetic_image = DiffAugment(synthetic_image, self.dsa_strategy, seed=seed, param=self.dsa_param)

                images_real_all.append(real_image)
                images_syn_all.append(synthetic_image)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)
            self.global_model.train()
            real_feature = self.global_model.embed(images_real_all).detach()
            self.global_model.eval()
            synthetic_feature = self.global_model.embed(images_syn_all)

            for i, c in enumerate(self.classes):
                mean_real_feature = torch.mean(real_feature[num_real_image[i] : num_real_image[i+1]], dim=0)
                mean_synthetic_feature = torch.mean(synthetic_feature[self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i+1]], dim=0)
                loss += torch.sum((mean_real_feature - mean_synthetic_feature)**2)

            total_loss += loss.item()
            optimizer_image.zero_grad()
            loss.backward()
            total_norm = nn.utils.clip_grad_norm_([self.synthetic_images,], max_norm=self.clip_norm)
            optimizer_image.step()

            if dc_iteration % 200 == 0 or dc_iteration == self.dc_iterations:
                logging.info(f'client {self.cid}, data condensation {dc_iteration}, total loss = {loss.item()}, avg loss = {loss.item() / len(self.classes)}')

        # return S_k
        synthetic_labels = torch.cat([torch.ones(self.ipc_dict[c]) * c for c in self.classes])
        # torch.save({'data': self.synthetic_images.detach().cpu(), 'label': synthetic_labels.detach().cpu()}, os.path.join(self.save_root_path, f"round{self.round}_client{self.cid}.pt"))
        return copy.deepcopy(self.synthetic_images.detach()), copy.deepcopy(synthetic_labels), total_loss/(len(self.classes)*self.dc_iterations), self.ipc_dict, self.accumulate_num_syn_imgs, prototypes, logit_prototypes

    def train_weighted_MMD(self):
        self.round += 1

        # initialize S_k and initialize optimizer
        self.initialization()
        optimizer_image = torch.optim.SGD([self.synthetic_images,], lr=self.image_lr, momentum=self.image_momentum, weight_decay=self.image_weight_decay)
        optimizer_image.zero_grad()
        loss_fn = nn.CrossEntropyLoss()
        logging.info(f"client {self.cid} have real samples {[len(self.train_set.indices_class[c]) for c in self.classes]}")
        logging.info(f"client {self.cid} will condense {self.ipc_dict} samples for each class it owns")
        
        if self.round == 0:
            self.global_model = get_model(self.model_name, self.dataset_info).to(self.device)
        prototypes = self.get_feature_prototype()
        logit_prototypes = self.get_logit_prototype()

        logging.info(f"loss weighted matching the samples")
        self.train_set.cal_loss(copy.deepcopy(self.global_model), copy.deepcopy(self.prev_global_model), lamda=self.lamda, gamma=self.gamma, b=self.b, rounds=self.round, cid=self.cid, save_root_path=self.save_root_path)
        self.train_set.pre_sample(it=self.dc_iterations+1, bs=self.real_batch_size)
        
        total_loss = 0.
        mmd_criterion = M3DLoss(kernel_type=self.kernel, device=self.device)
        self.global_model.train()
        for param in list(self.global_model.parameters()):
            param.requires_grad = False
        
        for dc_iteration in range(self.dc_iterations+1):
            loss = torch.tensor(0.0).to(self.device)
            images_real_all = []
            images_syn_all = []
            num_real_image = [0, ]
            for i, c in enumerate(self.classes):
                real_image = self.train_set.images_all[self.train_set.sample_indices[c][dc_iteration:dc_iteration+self.real_batch_size]]
                # real_image = self.train_set.weighted_sample(c, dc_iteration, self.real_batch_size)
                num_real_image.append(num_real_image[-1] + real_image.shape[0]) 
                synthetic_image = self.synthetic_images[self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i+1]].reshape(
                    (self.ipc_dict[c], self.dataset_info['channel'], self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))

                if self.dsa:
                    seed = int(time.time() * 1000) % 100000
                    real_image = DiffAugment(real_image, self.dsa_strategy, seed=seed, param=self.dsa_param)
                    synthetic_image = DiffAugment(synthetic_image, self.dsa_strategy, seed=seed, param=self.dsa_param)

                images_real_all.append(real_image)
                images_syn_all.append(synthetic_image)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)
            self.global_model.train()
            real_feature = self.global_model.embed(images_real_all).detach()
            self.global_model.eval()
            synthetic_feature = self.global_model.embed(images_syn_all)

            for i, c in enumerate(self.classes):
                loss += mmd_criterion(real_feature[num_real_image[i] : num_real_image[i+1]], synthetic_feature[self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i+1]])

            total_loss += loss.item()
            optimizer_image.zero_grad()
            loss.backward()
            total_norm = nn.utils.clip_grad_norm_([self.synthetic_images,], max_norm=self.clip_norm)
            optimizer_image.step()

            if dc_iteration % 200 == 0 or dc_iteration == self.dc_iterations:
                logging.info(f'client {self.cid}, data condensation {dc_iteration}, total loss = {loss.item()}, avg loss = {loss.item() / len(self.classes)}')

        # return S_k
        synthetic_labels = torch.cat([torch.ones(self.ipc_dict[c]) * c for c in self.classes])
        # torch.save({'data': self.synthetic_images.detach().cpu(), 'label': synthetic_labels.detach().cpu()}, os.path.join(self.save_root_path, f"round{self.round}_client{self.cid}.pt"))
        return copy.deepcopy(self.synthetic_images.detach()), copy.deepcopy(synthetic_labels), total_loss/(len(self.classes)*self.dc_iterations), self.ipc_dict, self.accumulate_num_syn_imgs, prototypes, logit_prototypes

    def get_feature_prototype(self):
        logging.info(f"get_feature_prototype")
        prototypes = {c: None for c in self.classes}
        self.global_model.eval()
        for param in list(self.global_model.parameters()):
            param.requires_grad = False
        for c in self.classes:
            tot_num_c = len(self.train_set.indices_class[c])
            if tot_num_c > 500:
                real_feature_c = []
                batch_size = 0
                for it in range(0, tot_num_c, 500):
                    if it + 500 >= tot_num_c:
                        real_feature_c_batch = self.global_model.embed(self.train_set.images_all[self.train_set.indices_class[c][it: tot_num_c]]).detach()
                        real_feature_c.append(torch.sum(real_feature_c_batch, dim=0))
                    else:
                        real_feature_c_batch = self.global_model.embed(self.train_set.images_all[self.train_set.indices_class[c][it: it+500]]).detach()
                        real_feature_c.append(torch.sum(real_feature_c_batch, dim=0))
                real_feature_c = torch.vstack(real_feature_c)
                real_feature_c = torch.sum(real_feature_c, dim=0) / tot_num_c
                prototypes[c] = (real_feature_c, tot_num_c)
                del real_feature_c
            else:
                real_images_c = self.train_set.get_all_images(c)
                real_feature_c = self.global_model.embed(real_images_c)
                prototypes[c] = (torch.mean(real_feature_c, dim=0), tot_num_c)
                del real_feature_c, real_images_c
            torch.cuda.empty_cache()

        return prototypes

    def get_logit_prototype(self):
        logging.info(f"get_logit_prototype")
        prototypes = {c: None for c in self.classes}
        self.global_model.eval()
        for param in list(self.global_model.parameters()):
            param.requires_grad = False
        for c in self.classes:
            tot_num_c = len(self.train_set.indices_class[c])
            if tot_num_c > 500:
                real_logit_c = []
                real_score_c = []
                for it in range(0, tot_num_c, 500):
                    if it + 500 >= tot_num_c:
                        real_logit_c_batch = self.global_model(self.train_set.images_all[self.train_set.indices_class[c][it: tot_num_c]]).detach()
                        real_logit_c_batch_sm = F.softmax(real_logit_c_batch, dim=1)
                        real_score_c.append(torch.log((real_logit_c_batch_sm[:, c]+1e-5) / (1 - real_logit_c_batch_sm[:, c]+1e-5)))
                        real_logit_c.append(torch.sum(real_logit_c_batch, dim=0))
                    else:
                        real_logit_c_batch = self.global_model(self.train_set.images_all[self.train_set.indices_class[c][it: it+500]]).detach()
                        real_logit_c_batch_sm = F.softmax(real_logit_c_batch, dim=1)
                        real_score_c.append(torch.log((real_logit_c_batch_sm[:, c]+1e-5) / (1 - real_logit_c_batch_sm[:, c]+1e-5)))
                        real_logit_c.append(torch.sum(real_logit_c_batch, dim=0))
                real_logit_c = torch.vstack(real_logit_c)
                real_score_c = torch.cat(real_score_c)
                real_logit_c = torch.sum(real_logit_c, dim=0) / tot_num_c
                prototypes[c] = (real_logit_c, tot_num_c)
                del real_logit_c
            else:
                real_images_c = self.train_set.get_all_images(c)
                real_logit_c = self.global_model(real_images_c)
                real_logit_c_sm = F.softmax(real_logit_c, dim=1)
                real_score_c = torch.log((real_logit_c_sm[:, c]+1e-5) / (1 - real_logit_c_sm[:, c]+1e-5))
                prototypes[c] = (torch.mean(real_logit_c, dim=0), tot_num_c)
                del real_logit_c, real_images_c
            torch.cuda.empty_cache()

        return prototypes

    def recieve_model(self, model_name, global_model=None):
        self.model_name = model_name
        if global_model is not None:
            if self.round == -1:
                self.prev_global_model = copy.deepcopy(global_model)
            else:
                self.prev_global_model = copy.deepcopy(self.global_model)
            self.global_model = copy.deepcopy(global_model)
            self.global_model.eval()

    def initialization(self):
        if self.init == 'real':
            logging.info("initialized by real images")
            for i, c in enumerate(self.classes):
                self.synthetic_images.data[self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i+1]] = self.train_set.get_images(c, self.ipc_dict[c], avg=False).detach().data
        elif self.init == 'real_avg':
            logging.info("initialized by average real images")
            for i, c in enumerate(self.classes):
                self.synthetic_images.data[self.accumulate_num_syn_imgs[i] : self.accumulate_num_syn_imgs[i+1]] = self.train_set.get_images(c, self.ipc_dict[c], avg=True).detach().data
        elif self.init == 'random_noise':
            logging.info("initialized by random noise")
            pass
