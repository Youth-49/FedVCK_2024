import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import copy
import random
import math
import logging
from torchvision.utils import save_image
import os
from medmnist import PathMNIST, OCTMNIST, OrganSMNIST, OrganCMNIST, PneumoniaMNIST, RetinaMNIST


def get_dataset(dataset, dataset_root, batch_size):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.MNIST(dataset_root, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.MNIST(dataset_root, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.CIFAR10(dataset_root, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.CIFAR10(dataset_root, train=False, download=True, transform=transform)
        class_names = trainset.classes
    elif dataset == 'STL':
        channel = 3
        im_size = (96, 96)
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.STL10(dataset_root, split='train', download=True, transform=transform)  # no augmentation
        testset = datasets.STL10(dataset_root, split='test', download=True, transform=transform)
        class_names = None
    elif dataset == 'STL32':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.STL10(dataset_root, split='train', download=True, transform=transform)  # no augmentation
        testset = datasets.STL10(dataset_root, split='test', download=True, transform=transform)
        class_names = None
    elif dataset == 'PathMNIST':
        channel = 3
        im_size = (28, 28)
        num_classes = 9
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = PathMNIST(split="train", download=True, root=dataset_root, transform=transform) # no transformation
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = PathMNIST(split="test", download=True, root=dataset_root, transform=transform)
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    elif dataset == 'OrganSMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 11
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = OrganSMNIST(split="train", download=True, root=dataset_root, transform=transform) # no transformation
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = OrganSMNIST(split="test", download=True, root=dataset_root, transform=transform)
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    elif dataset == 'OCTMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 4
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = OCTMNIST(split="train", download=True, root=dataset_root, transform=transform) # no transformation
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = OCTMNIST(split="test", download=True, root=dataset_root, transform=transform)
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    elif dataset == 'ImageNette':
        from fastai.vision.all import untar_data, URLs
        channel = 3
        num_classes = 10
        im_size = (64, 64)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        path = untar_data(URLs.IMAGENETTE)
        print(path)
        transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.ImageFolder(root=f'{path}/train', transform=transform) # cancel augment
        testset = datasets.ImageFolder(root=f'{path}/val', transform=transform)
        trainset.labels = np.array(np.squeeze(trainset.targets).tolist(), dtype='int64')
        testset.labels = np.array(np.squeeze(testset.targets).tolist(), dtype='int64')
        class_names = range(10)
    elif dataset == 'OrganCMNIST224':
        channel = 1
        num_classes = 11
        im_size = (224, 224)
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = OrganCMNIST(split="train", download=True, root=dataset_root, transform=transform, size=224) # no transformation
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = OrganCMNIST(split="test", download=True, root=dataset_root, transform=transform, size=224)
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    elif dataset == 'PneumoniaMNIST224':
        channel = 1
        num_classes = 2
        im_size = (224, 224)
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = PneumoniaMNIST(split="train", download=True, root=dataset_root, transform=transform, size=224) # no transformation
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = PneumoniaMNIST(split="test", download=True, root=dataset_root, transform=transform, size=224) # no transformation
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    elif dataset == 'RetinaMNIST224':
        channel = 3
        num_classes = 5
        im_size = (224, 224)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = RetinaMNIST(split="train", download=True, root=dataset_root, transform=transform, size=224) # no transformation
        trainset.labels = np.array(np.squeeze(trainset.labels).tolist(), dtype='int64')
        testset = RetinaMNIST(split="test", download=True, root=dataset_root, transform=transform, size=224) # no transformation
        testset.labels = np.array(np.squeeze(testset.labels).tolist(), dtype='int64')
        class_names = trainset.info['label'].values()
    else:
        exit(f'unknown dataset: {dataset}')

    dataset_info = {
        'name': dataset,
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'classes_names': class_names,
        'mean': mean,
        'std': std,
    }

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                  num_workers=0)  # pin memory

    return dataset_info, trainset, testset, testloader

class PerLabelDatasetNonIID():
    def __init__(self, dst_train, classes, channel, device):  # images: n x c x h x w tensor
        self.images_all = []
        self.labels_all = []
        self.indices_class = {c: [] for c in classes}
        self.device = device

        self.images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        self.labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(self.labels_all):
            if lab not in classes:
                continue
            self.indices_class[lab].append(i)
        if len(self.images_all) > 0:
            self.images_all = torch.cat(self.images_all, dim=0).to(device)
            self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=device)
            self.loss_all = None
            self.sorted_indices_class = {c: [] for c in classes}
            self.sample_prob = {c: [] for c in classes}
            self.sample_indices = {c: [] for c in classes}

    def __len__(self):
        return self.images_all.shape[0]

    def get_random_images(self, n):  # get n random images
        idx_shuffle = np.random.permutation(range(self.images_all.shape[0]))[:n]
        return self.images_all[idx_shuffle]

    def get_images(self, c, n, avg=False):  # get n random images from class c
        if not avg:
            if len(self.indices_class[c]) >= n:
                idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            else:
                # sampled_idx = np.random.choice(self.indices_class[c], n - len(self.indices_class[c]), replace=True)
                # idx_shuffle = np.concatenate((self.indices_class[c], sampled_idx), axis=None)
                idx_shuffle = self.indices_class[c]
            return self.images_all[idx_shuffle]
        else:
            sampled_imgs = []
            for _ in range(n):
                if len(self.indices_class[c]) >= 5:
                    idx = np.random.choice(self.indices_class[c], 5, replace=False)
                else:
                    idx = np.random.choice(self.indices_class[c], 5, replace=True)
                sampled_imgs.append(torch.mean(self.images_all[idx], dim=0, keepdim=True))
            sampled_imgs = torch.cat(sampled_imgs, dim=0).to(self.device)
            return sampled_imgs
        
    def get_all_images(self, c):
        all_images = self.images_all[self.indices_class[c]]
        return all_images
        
    def sort_image_by_model(self, model, thres=0.5, rounds=None, cid=None, save_root_path=None):
        loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        for i, c in enumerate(self.indices_class.keys()):
            all_images_c = self.get_all_images(c)
            all_labels_c = torch.ones(all_images_c.shape[0])*c
            all_labels_c = all_labels_c.long().to(self.device)
            model.eval()
            with torch.no_grad():
                all_pred_c = model(all_images_c)
                loss = loss_function(all_pred_c, all_labels_c)
                sorted_loss, sorted_indices = torch.sort(loss, descending=True, dim=0)

            thres = int(math.ceil(len(sorted_indices) * thres))
            # logging.info(f"{sorted_loss[:thres]}")
            self.sorted_indices_class[c] = [self.indices_class[c][idx] for idx in sorted_indices[:thres]]
            # save_image(self.images_all[self.sorted_indices_class[c]].data.clone(), os.path.join(save_root_path, f'hard_imgs{rounds}_{cid}_{c}.png'), normalize=True, scale_each=True, nrow=10)
            del all_images_c, all_labels_c, all_pred_c
            torch.cuda.empty_cache()

    def cal_loss(self, model, prev_model, lamda=0.5, gamma=1.0, b=0.7, rounds=None, cid=None, save_root_path=None):
        loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        model.eval()
        prev_model.eval()
        with torch.no_grad():
            if self.images_all.shape[0] > 500:
                all_preds = []
                all_preds_prev = []
                batch_size = 500
                total_num = self.images_all.shape[0]
                for idx in range(0, total_num, batch_size):
                    batch_st = idx 
                    if batch_st + batch_size >= total_num:
                        batch_ed = total_num
                    else:
                        batch_ed = idx+batch_size
                    all_preds.append(model(self.images_all[batch_st: batch_ed]))
                    all_preds_prev.append(prev_model(self.images_all[batch_st: batch_ed]))
                all_preds = torch.cat(all_preds, dim=0)
                all_preds_prev = torch.cat(all_preds_prev, dim=0)
                all_preds = (1-lamda) * all_preds + lamda * all_preds_prev
                print(all_preds.shape)
            else:
                all_preds = model(self.images_all)
                all_preds_prev = prev_model(self.images_all)
                all_preds = (1-lamda) * all_preds + lamda * all_preds_prev
            self.loss_all = loss_function(all_preds, self.labels_all).type(torch.float64)
            # self.loss_all = 1.0/(1.0+torch.exp(-gamma * (self.loss_all-0.7))).cpu()
            self.loss_all = 1.0/(1.0+torch.exp(-gamma * (self.loss_all-b))).cpu()
            # logging.info(f"{self.loss_all.cpu().tolist()}")
        del all_preds
        torch.cuda.empty_cache()

    def norm_loss(self):
        for i, c in enumerate(self.indices_class.keys()):
            self.sample_prob[c] = F.softmax(self.loss_all[self.indices_class[c]], dim=0)
            hist, _ = np.histogram(self.sample_prob[c], bins=10)
            logging.info(f"class {c} have {len(self.indices_class[c])} samples, histogram: {hist}")
    
    def pre_sample(self, it, bs):
        for i, c in enumerate(self.indices_class.keys()):
            self.sample_prob[c] = F.softmax(self.loss_all[self.indices_class[c]], dim=0)
            self.sample_indices[c] = np.random.choice(self.indices_class[c], size=it*bs, replace=True, p=self.sample_prob[c])
            hist, bin_edges = np.histogram(self.sample_prob[c], bins=10)
            logging.info(f"class {c} have {len(self.indices_class[c])} samples, histogram: {hist}, bin edged: {bin_edges}")

    def weighted_sample(self, c, it, bs):
        return self.images_all[self.sample_indices[c][it:it+bs]]

    def get_images_loss(self, c, n):
        if len(self.indices_class[c]) >= n:
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        else:
            idx_shuffle = self.indices_class[c]
        return self.images_all[idx_shuffle], self.loss_all[idx_shuffle]

    def get_sorted_images(self, c, n):
        if len(self.sorted_indices_class[c]) >= n:
            idx_shuffle = np.random.permutation(self.sorted_indices_class[c])[:n]
        else:
            idx_shuffle = self.sorted_indices_class[c]
        return self.images_all[idx_shuffle]