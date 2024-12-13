import copy
import os
import random
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np
from torchvision.utils import save_image

from src.client import Client
from src.utils import DiffAugment, ParamDiffAug, MMDLoss, ContrastiveLoss, SupervisedContrastiveLoss
from .models import Projector

import matplotlib.pyplot as plt 
import torch.nn.functional as F
import json
import logging

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

def get_embedding(model, data_input, device, batch_size=1024, detach=True):
    embedding_list = []
    total_num = data_input.shape[0]
    for idx in range(0, total_num, batch_size):
        batch_st = idx 
        if batch_st + batch_size >= total_num:
            batch_ed = total_num
        else:
            batch_ed = idx+batch_size
        if detach:
            embedding_list.append(model.embed(data_input[batch_st: batch_ed]).detach())
        else:
            embedding_list.append(model.embed(data_input[batch_st: batch_ed]))
    
    print(f"get embedding > 5000 with batchsize={batch_size}...")
    get_gpu_mem_info(device)
    embedding_list = torch.cat(embedding_list, dim=0)
    return embedding_list

class Server:
    def __init__(
        self,
        train_set,
        ipc, 
        dataset_info,
        global_model_name: str,
        global_model: nn.Module,
        clients: list[Client],
        # --- model training params ---
        communication_rounds: int,
        join_ratio: float,
        batch_size: int,
        model_epochs: int,
        lr_server: float,
        momentum_server: float,
        weight_decay_server: float,
        lr_head: float,
        momentum_head: float, 
        weight_decay_head: float,
        weighted_matching: bool,
        weighted_sample: bool,
        weighted_mmd: bool,
        contrastive_way: str,
        con_beta: float,
        con_temp: float, 
        topk: int, 
        dsa: bool,
        dsa_strategy: str,
        preserve_all: bool,
        # --- test and evaluation information ---
        eval_gap: int,
        test_set: object,
        test_loader: DataLoader,
        device: torch.device,
        # --- save model and synthetic images ---
        model_identification: str,
        save_root_path: str
    ):
        self.train_set = train_set
        self.ipc = ipc
        self.dataset_info = dataset_info
        self.global_model_name = global_model_name
        self.global_model = global_model.to(device)
        self.clients = clients

        self.communication_rounds = communication_rounds
        self.join_ratio = join_ratio
        self.batch_size = batch_size
        self.model_epochs = model_epochs
        self.lr_server = lr_server
        self.momentum_server = momentum_server
        self.weight_decay_server = weight_decay_server
        self.lr_head = lr_head
        self.momentum_head = momentum_head
        self.weight_decay_head = weight_decay_head
        self.weighted_matching = weighted_matching
        self.weighted_sample = weighted_sample
        self.weighted_mmd = weighted_mmd
        self.contrastive_way = contrastive_way
        self.con_beta = con_beta
        self.con_temp = con_temp
        self.topk = topk
        self.dsa = dsa
        self.dsa_strategy = dsa_strategy
        self.dsa_param = ParamDiffAug()
        self.preserve_all = preserve_all

        self.eval_gap = eval_gap
        self.test_set = test_set
        self.test_loader = test_loader
        self.device = device

        self.model_identification = model_identification
        self.save_root_path = save_root_path

    def fit(self):
        evaluate_acc = 0
        round_list = []
        evaluate_acc_list = []
        img_syn_loss = {idx: [] for idx in range(len(self.clients))}

        all_synthetic_data = []
        all_synthetic_label = []
        all_syn_imgs_c = {c: [] for c in range(0, self.dataset_info['num_classes'])}
        mmd_gap = {c: [] for c in range(0, self.dataset_info['num_classes'])}
        accumlate_mmd = {c: [] for c in range(0, self.dataset_info['num_classes'])}
        prev_syn_proto = None

        for rounds in range(self.communication_rounds):
            logging.info(f' ====== round {rounds} ======')
            start_time = time.time()
            logging.info('---------- client training ----------')

            selected_clients = self.select_clients()
            selected_clients_id = [selected_client.cid for selected_client in selected_clients]
            logging.info(f'selected clients: {selected_clients_id}')

            server_prototypes = {c: 0 for c in range(0, self.dataset_info['num_classes'])}
            server_proto_tensor = []
            server_logit_prototypes = {c: 0 for c in range(0, self.dataset_info['num_classes'])}
            server_logit_proto_tensor = []
            
            num_samples = {c: 0 for c in range(0, self.dataset_info['num_classes'])}
            syn_imgs_all = {c: [] for c in range(0, self.dataset_info['num_classes'])}
            syn_imgs_num_cur = {c: 0 for c in range(0, self.dataset_info['num_classes'])}
            idx_client = {c: {client.cid: [] for client in selected_clients} for c in range(0,self.dataset_info['num_classes'])}
            for client in selected_clients:
                print(f"Round {rounds}, client {client.cid} start training...")
                get_gpu_mem_info(self.device)
                client.recieve_model(self.global_model_name, self.global_model)
                # if len(client.classes) == 0:
                #     logging.info(f"skip client {client.cid}")
                #     continue
                condense_st_time = time.time()
                if self.weighted_sample:
                    imgs, labels, syn_loss, ipc_dict, accmulate_num_syn_imgs, prototypes, logit_prototypes = client.train_weighted_sample()
                elif self.weighted_mmd:
                    imgs, labels, syn_loss, ipc_dict, accmulate_num_syn_imgs, prototypes, logit_prototypes = client.train_weighted_MMD()
                condense_ed_time = time.time()
                logging.info(f"Round {rounds}, client {client.cid} condense time: {condense_ed_time - condense_st_time}")

                img_syn_loss[client.cid].append(syn_loss)
                for i, c in enumerate(client.classes):
                    synthetic_image_c = imgs[accmulate_num_syn_imgs[i] : accmulate_num_syn_imgs[i+1]].reshape(
                        (ipc_dict[c], self.dataset_info['channel'], self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))
                    syn_imgs_all[c].append(synthetic_image_c)
                    idx_client[c][client.cid] = range(syn_imgs_num_cur[c], syn_imgs_num_cur[c] + ipc_dict[c])
                    syn_imgs_num_cur[c] += ipc_dict[c]
                
                for i, c in enumerate(client.classes):
                    logging.info(f"client {client.cid}, class {c} have {prototypes[c][1]} samples")
                    server_prototypes[c] += prototypes[c][0] * prototypes[c][1]
                    num_samples[c] += prototypes[c][1]
                    server_logit_prototypes[c] += logit_prototypes[c][0] * logit_prototypes[c][1]
                
                print(f"Round {rounds}, client {client.cid} finish training...")
                get_gpu_mem_info(self.device)

            logging.info(f"server receives {syn_imgs_num_cur} condensed samples for each class")

            for c in range(self.dataset_info['num_classes']):
                server_prototypes[c] /= num_samples[c]
                server_logit_prototypes[c] /= num_samples[c]
                server_proto_tensor.append(server_prototypes[c])
                server_logit_proto_tensor.append(server_logit_prototypes[c])

            server_proto_tensor = torch.vstack(server_proto_tensor).to(self.device).detach()
            server_proto_tensor = F.normalize(server_proto_tensor, dim=1) # 是不是应该先norm再平均
            server_logit_proto_tensor = torch.vstack(server_logit_proto_tensor).to(self.device).detach()
            logging.info(f"logit_proto before softmax: {server_logit_proto_tensor}")
            _, relation_class = self.get_mask(server_logit_proto_tensor, k = self.topk)
            if rounds > 0:
                for c in range(self.dataset_info['num_classes']):
                    if c not in relation_class[c]:
                        logging.info(f"class {c} not in relation_class, manually added")
                        relation_class[c][-1] = c
            
            logging.info(f"shape of prototypes in tensor: {server_proto_tensor.shape}")
            logging.info(f"shape of logit prototypes in tensor: {server_logit_proto_tensor.shape}")
            logging.info(f"relation tensor: {relation_class}")

            for c in range(0, self.dataset_info['num_classes']):
                syn_imgs_all[c] = torch.vstack(syn_imgs_all[c])

            synthetic_data = []
            synthetic_label = []
            for c in range(0, self.dataset_info['num_classes']):
                all_syn_imgs_c[c].append(syn_imgs_all[c])
                synthetic_data.append(syn_imgs_all[c])
                synthetic_label.append(torch.ones(syn_imgs_all[c].shape[0])*c)

            synthetic_data = torch.vstack(synthetic_data)
            synthetic_label = torch.cat(synthetic_label, dim=0)

            logging.info('---------- update global model ----------')
            # update model parameters by SGD
            all_synthetic_data.append(synthetic_data)
            all_synthetic_label.append(synthetic_label)
            logging.info(len(synthetic_data))

            preserve_thres = self.communication_rounds // 2
            logging.info(f"preserve threshold: {preserve_thres}")
            if (not self.preserve_all) and (len(all_synthetic_data) > preserve_thres):
                all_synthetic_data = all_synthetic_data[-preserve_thres: ]
                all_synthetic_label = all_synthetic_label[-preserve_thres: ]

            logging.info(len(all_synthetic_data))
            all_synthetic_data_eval = torch.cat(all_synthetic_data, dim=0).cpu()
            all_synthetic_label_eval = torch.cat(all_synthetic_label, dim=0).cpu()
            synthetic_dataset = TensorDataset(all_synthetic_data_eval, all_synthetic_label_eval)
            logging.info(f"Round {rounds}: # synthetic sample: {len(synthetic_dataset)}")
            synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=2)

            self.global_model.train()
            model_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=self.lr_server,
                weight_decay=self.weight_decay_server,
                momentum=self.momentum_server,
            )
            model_optimizer.zero_grad()
            lr_schedule = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=(self.model_epochs//2), gamma=0.1)
            loss_function = torch.nn.CrossEntropyLoss()
            z_dim = server_proto_tensor.shape[1]
            relation_sup_con_loss = SupervisedContrastiveLoss(num_classes=self.dataset_info['num_classes'], device=self.device, temperature=self.con_temp, z_dim=z_dim, relation_class=relation_class)
            # con_loss = ContrastiveLoss(z_dim, device=self.device, temperature=self.con_temp)
            mlp_head_optimizer = torch.optim.Adam(
                relation_sup_con_loss.head.parameters(),
                lr=self.lr_head,
                weight_decay=self.weight_decay_head,
                # momentum=self.momentum_head,
            )
            mlp_head_optimizer.zero_grad()
            head_lr_schedule = torch.optim.lr_scheduler.StepLR(mlp_head_optimizer, step_size=(self.model_epochs//2), gamma=0.1)

            print(f"Round {rounds}, global model start training...")
            get_gpu_mem_info(self.device)

            # evaluate ahead training
            acc, test_loss = self.evaluate()
            logging.info(f'round {rounds} evaluation: test acc is {acc:.4f}, test loss = {test_loss:.6f}')
            self.global_model.train()
            for param in list(self.global_model.parameters()):
                param.requires_grad = True
            for epoch in range(self.model_epochs+1):
                total_loss = 0
                total_con_loss = 0
                total_sample = 0
                for x, target in synthetic_dataloader:
                    n_sample = target.shape[0]
                    x, target = x.to(self.device), target.to(self.device) 
                    if self.con_beta > 0.:
                        features, _ = self.global_model(x, train=True)                   
                    if self.dsa:
                        x = DiffAugment(x, self.dsa_strategy, param=self.dsa_param)
 
                    target = target.long()
                    _, pred = self.global_model(x, train=True)
                    loss = loss_function(pred, target)
                    total_loss += loss.item() * n_sample

                    if self.con_beta > 0. and rounds > 0 and x.shape[0] > 1:
                        if self.contrastive_way == 'supcon_asym_syn':
                            assert prev_syn_proto is not None
                            positive_proto = prev_syn_proto[target, :]
                            loss_con = relation_sup_con_loss(features, target, positive_proto, asymmetric=True)
                        total_con_loss += loss_con.item() * n_sample
                        loss += self.con_beta * loss_con

                    model_optimizer.zero_grad()
                    loss.backward()
                    model_optimizer.step()
                    total_sample += n_sample
                    if self.con_beta > 0. and self.contrastive_way in ['supcon_asym', 'supcon_asym_syn'] and rounds > 0:
                        mlp_head_optimizer.step()

                total_loss /= total_sample
                total_con_loss /= total_sample
                # if self.dataset_info['name'] not in ['OCTMNIST']:
                lr_schedule.step()
                if self.con_beta > 0. and self.contrastive_way in ['supcon_asym', 'supcon_asym_syn'] and rounds > 0:
                    head_lr_schedule.step()
                if epoch == (self.model_epochs // 2):
                    logging.info(f"At epoch {epoch}, decay the con_beta with 0.1 factor")
                    self.con_beta *= 0.1

                if epoch%100 == 0 or epoch == self.model_epochs:
                    acc, test_loss = self.evaluate()
                    self.global_model.train()
                    logging.info(f"epoch {epoch}, train loss avg now = {total_loss:.6f}, train contrast loss now = {total_con_loss:.6f}, test acc now = {acc:.4f}, test loss now = {test_loss:.6f}")

            round_time = time.time() - start_time
            logging.info(f'epoch avg loss = {total_loss / self.model_epochs}, total time = {round_time}')
            
            print(f"Round {rounds}, global model finish training...")
            get_gpu_mem_info(self.device)

            logging.info(f"Round {rounds} finish, update the prev_syn_proto")
            prev_syn_proto = torch.zeros_like(server_proto_tensor).to(self.device)
            self.global_model.eval()
            with torch.no_grad():
                for c in range(0, self.dataset_info['num_classes']):
                    all_syn_cat = torch.cat(all_syn_imgs_c[c], dim=0)
                    logging.info(f"{all_syn_cat.shape}")
                    if all_syn_cat.shape[0] > 128:
                        for it in range(0, all_syn_cat.shape[0], 128):
                            if it + 128 >= all_syn_cat.shape[0]:
                                prev_syn_proto[c, :] += torch.sum(self.global_model.embed(all_syn_cat[it: ]).detach(), dim=0)
                            else:
                                prev_syn_proto[c, :] += torch.sum(self.global_model.embed(all_syn_cat[it: it+128]).detach(), dim=0)
                        prev_syn_proto[c, :] /= all_syn_cat.shape[0]
                    else:
                        prev_syn_proto[c, :] = torch.mean(self.global_model.embed(all_syn_cat).detach(), dim=0)
                prev_syn_proto = F.normalize(prev_syn_proto, dim=1).detach()
                logging.info(f"shape of prev_syn_proto: {prev_syn_proto.shape}")

            if rounds % self.eval_gap == 0:
                acc, test_loss = self.evaluate()
                logging.info(f'round {rounds} evaluation: test acc is {acc:.4f}, test loss = {test_loss:.6f}')
                evaluate_acc = acc
                round_list.append(rounds)
                evaluate_acc_list.append(evaluate_acc)

            # self.save_model(path=save_root_path, rounds=rounds, include_image=False)
            # torch.save(self.global_model.state_dict(), os.path.join(self.save_root_path, f"model_{rounds}.pt"))

        logging.info(evaluate_acc_list)
        logging.info(img_syn_loss)
        logging.info(mmd_gap)
        logging.info(accumlate_mmd)

    def get_mask(self, matrix, k=3, largest=True):
        min_val, min_idx = torch.topk(matrix, k=k, dim=-1, largest=largest)
        mask = torch.zeros_like(matrix)
        rows = torch.arange(min_idx.size(0)).unsqueeze(1)
        mask[rows, min_idx] = 1
        mask = mask.bool()
        return mask, min_idx

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def evaluate(self):
        prediction_matrix = {c: {c: 0 for c in range(self.dataset_info['num_classes'])} for c in range(self.dataset_info['num_classes'])}
        self.global_model.eval()
        with torch.no_grad():
            correct, total, test_loss = 0, 0, 0.
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = self.global_model(x)
                test_loss += F.cross_entropy(pred, target, reduction='sum').item()
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                for i in range(target.shape[0]):
                    prediction_matrix[target[i].item()][pred_label[i].item()] += 1

        logging.info(f"{prediction_matrix}")
        return correct / float(total), test_loss / float(total)
    
    def evaluate_model(self, model):
        model.eval()
        with torch.no_grad():
            correct, total, test_loss = 0, 0, 0.
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = model(x)
                test_loss += F.cross_entropy(pred, target, reduction='sum').item()
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        return correct / float(total), test_loss / float(total)

    def make_checkpoint(self, rounds):
        checkpoint = {
            'current_round': rounds,
            'model': self.global_model.state_dict()
        }
        return checkpoint

    def save_model(self, path, rounds, include_image):
        # torch.save(self.make_checkpoint(rounds), os.path.join(path, f'model_{rounds}.pt'))
        torch.save(self.make_checkpoint(rounds), os.path.join(path, 'model.pt'))
        if include_image:
            raise NotImplemented('not implement yet')