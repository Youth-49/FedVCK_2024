import argparse
import json
import os

import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from medmnist import PathMNIST, OCTMNIST, OrganSMNIST, OrganCMNIST, RetinaMNIST, PneumoniaMNIST
from fastai.vision.all import untar_data, URLs
import torch


def plot_client_data_distribution(num_classes, num_users, dict_users, labels, save_path):
    for client_id in dict_users.keys():
        print(len(dict_users[client_id]))

    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(num_classes)]
    for client_id, client_data in dict_users.items():
        for idx in client_data:
            label_distribution[labels[idx]].append(client_id)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, num_users + 1.5, 1),
             label=range(num_classes), rwidth=0.5)
    plt.xticks(np.arange(num_users), ["Client %d" %
                                      c_id for c_id in range(num_users)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    plt.title("Label Distribution on Different Clients")
    plt.savefig(save_path)

def partition(args):
    np.random.seed(args.seed)

    # prepare datasets for then partition latter
    if args.dataset == 'MNIST':
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.MNIST(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = [str(c) for c in range(num_classes)]
    elif args.dataset == 'CIFAR10':
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    elif args.dataset == 'FMNIST':
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.FashionMNIST(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    elif args.dataset == 'STL':
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.STL10(args.dataset_root, split='train', download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    elif args.dataset == 'STL32':
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.STL10(args.dataset_root, split='train', download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    elif args.dataset == 'PathMNIST':
        num_classes = 9
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = PathMNIST(split="train", download=True, root=args.dataset_root, transform=transform) # no transformation
        dataset.labels = np.array(np.squeeze(dataset.labels).tolist(), dtype='int64')
        class_names = dataset.info['label'].values()
        for lbl in dataset.info['label'].keys():
            print(f'class {lbl} have {(dataset.labels == int(lbl)).sum()} samples')
    elif args.dataset == 'OrganSMNIST':
        num_classes = 11
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = OrganSMNIST(split="train", download=True, root=args.dataset_root, transform=transform) # no transformation
        dataset.labels = np.array(np.squeeze(dataset.labels).tolist(), dtype='int64')
        class_names = dataset.info['label'].values()
        for lbl in dataset.info['label'].keys():
            print(f'class {lbl} have {(dataset.labels == int(lbl)).sum()} samples')
    elif args.dataset == 'OCTMNIST':
        num_channels = 1
        num_classes = 4
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = OCTMNIST(split="train", download=True, root=args.dataset_root, transform=transform) # no transformation
        dataset.labels = np.array(np.squeeze(dataset.labels).tolist(), dtype='int64')
        for lbl in dataset.info['label'].keys():
            print(f'class {lbl} have {(dataset.labels == int(lbl)).sum()} samples')
    elif args.dataset == 'ImageNette':
        num_channels = 3
        num_classes = 10
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        path = untar_data(URLs.IMAGENETTE)
        print(path)
        transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.ImageFolder(root=f'{path}/train', transform=transform) # cancel augment
        dataset.labels = np.array(np.squeeze(dataset.targets).tolist(), dtype='int64')
        for lbl in range(10):
            print(f'class {lbl} have {(dataset.labels == int(lbl)).sum()} samples')
    elif args.dataset == 'OrganCMNIST224':
        num_classes = 11
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = OrganCMNIST(split="train", download=True, root=args.dataset_root, transform=transform, size=224) # no transformation
        dataset.labels = np.array(np.squeeze(dataset.labels).tolist(), dtype='int64')
        class_names = dataset.info['label'].values()
        for lbl in dataset.info['label'].keys():
            print(f'class {lbl} have {(dataset.labels == int(lbl)).sum()} samples')
    elif args.dataset == 'PneumoniaMNIST224':
        num_classes = 2
        mean = [0.5,]
        std = [0.5,]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = PneumoniaMNIST(split="train", download=True, root=args.dataset_root, transform=transform, size=224) # no transformation
        dataset.labels = np.array(np.squeeze(dataset.labels).tolist(), dtype='int64')
        class_names = dataset.info['label'].values()
        for lbl in dataset.info['label'].keys():
            print(f'class {lbl} have {(dataset.labels == int(lbl)).sum()} samples')
    elif args.dataset == 'RetinaMNIST224':
        num_classes = 5
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = RetinaMNIST(split="train", download=True, root=args.dataset_root, transform=transform, size=224) # no transformation
        dataset.labels = np.array(np.squeeze(dataset.labels).tolist(), dtype='int64')
        class_names = dataset.info['label'].values()
        for lbl in dataset.info['label'].keys():
            print(f'class {lbl} have {(dataset.labels == int(lbl)).sum()} samples')
    else:
        exit(f'unknown dataset: f{args.dataset}')

    if args.dataset in ['CIFAR10', 'CIFAR100', 'FMNIST', 'CIFAR100C']:
        labels = np.array(dataset.targets, dtype='int64')
    elif args.dataset in ['PathMNIST', 'OrganAMNIST', 'OCTMNIST','OrganSMNIST', 'ImageNette', 'OrganCMNIST224', 'PneumoniaMNIST224', 'RetinaMNIST224', 'STL', 'STL32']:
        labels = dataset.labels

    dict_users = {}
    dict_classes = {}

    def dirichlet_split():
        min_size = -1
        min_require_size = 0
        K = num_classes
        if args.dataset in ['CIFAR10', 'FMNIST']:
            labels = np.array(dataset.targets, dtype='int64')
        elif args.dataset in ['PathMNIST', 'OCTMNIST', 'OrganSMNIST', 'OrganCMNIST', 'ImageNette', 'OrganCMNIST224', 'PneumoniaMNIST224', 'RetinaMNIST224', 'STL', 'STL32']:
            labels = dataset.labels
        N = labels.shape[0]
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.client_num)]
            for k in range(K):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.alpha, args.client_num))
                # print(proportions)
                proportions = np.array([p * (len(idx_j) < N / args.client_num) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                print(min_size)

        for j in range(args.client_num):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]

        return dict_users

    def label_split():
        if args.dataset in ['CIFAR10', 'FMNIST']:
            labels = np.array(dataset.targets, dtype='int64')
        elif args.dataset in ['PathMNIST', 'OCTMNIST', 'OrganSMNIST', 'OrganCMNIST', 'ImageNette', 'OrganCMNIST224', 'PneumoniaMNIST224', 'RetinaMNIST224', 'STL', 'STL32']:
            labels = dataset.labels
        times = [0 for i in range(num_classes)]
        contain = []
        for i in range(args.client_num):
            current = [i % num_classes]
            times[i % num_classes] += 1
            j = 1
            while (j < args.num_classes_per_client):
                ind = np.random.randint(0, num_classes-1)
                if (ind not in current):
                    j = j+1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)

        dict_users = {i: np.ndarray(0,dtype=np.int64) for i in range(args.client_num)}
        for i in range(num_classes):
            idx_k = np.where(labels == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(args.client_num):
                if i in contain[j]:
                    dict_users[j] = np.append(dict_users[j], split[ids])
                    ids+=1

        for client_id in dict_users.keys():
            dict_users[client_id] = dict_users[client_id].tolist()
        return dict_users

        
    def pathological_split():
        if args.dataset in ['CIFAR10', 'FMNIST']:
            labels = np.array(dataset.targets, dtype='int64')
        elif args.dataset in ['PathMNIST', 'OCTMNIST', 'OrganSMNIST', 'OrganCMNIST', 'ImageNette', 'OrganCMNIST224', 'PneumoniaMNIST224', 'RetinaMNIST224', 'STL', 'STL32']:
            labels = dataset.labels
        num_samples = labels.shape[0]
        num_shards = args.num_classes_per_client * args.client_num
        assert num_samples % num_shards == 0
        num_imgs_per_shard = int(num_samples / num_shards)
        print(f"total sample: {num_samples}, num_shards: {num_shards}, num_imgs_per_shard: {num_imgs_per_shard}")

        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([], dtype='int64') for i in range(args.client_num)}
        idxs = np.arange(num_samples)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
        idxs = idxs_labels[0,:]

        # divide and assign
        for i in range(args.client_num):
            rand_set = set(np.random.choice(idx_shard, args.num_classes_per_client, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs_per_shard:(rand+1)*num_imgs_per_shard]), axis=0)
        
        for client_id in dict_users.keys():
            dict_users[client_id] = dict_users[client_id].tolist()
        return dict_users


    if args.method == 'dirichlet':
        dict_users = dirichlet_split()
    elif args.method == 'label':
        dict_users = label_split()
    elif args.method == 'pathological':
        dict_users = pathological_split()

    net_cls_counts = {}

    for net_i, dataidx in dict_users.items():
        dict_classes[net_i] = []
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        for c, cnt in tmp.items():
            if cnt >= 10:
                dict_classes[net_i].append(int(c))

    print('Data statistics: %s' % str(net_cls_counts))

    save_path = os.path.join(os.path.dirname(__file__), '../', 'split_file')
    if args.method == 'dirichlet':
        file_name = f'{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
        plot_client_data_distribution(num_classes, args.client_num, dict_users, labels, save_path=f'{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.png')
    elif args.method == 'label':
        file_name = f'{args.dataset}_client_num={args.client_num}_label={args.num_classes_per_client}.json'
        plot_client_data_distribution(num_classes, args.client_num, dict_users, labels, save_path=f'{args.dataset}_client_num={args.client_num}_label={args.num_classes_per_client}.png')
    elif args.method == 'pathological':
        file_name = f'{args.dataset}_client_num={args.client_num}_pathological={args.num_classes_per_client}.json'
        plot_client_data_distribution(num_classes, args.client_num, dict_users, labels, save_path=f'{args.dataset}_client_num={args.client_num}_pathological={args.num_classes_per_client}.png')

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, file_name), 'w') as json_file:
        json.dump({
            "client_idx": [dict_users[i] for i in range(args.client_num)],
            "client_classes": [dict_classes[i] for i in range(args.client_num)],
        }, json_file, indent=4)

if __name__ == "__main__":
    partition_parser = argparse.ArgumentParser()

    partition_parser.add_argument("--dataset", type=str, default='CIFAR10')
    partition_parser.add_argument("--method", type=str, default='dirichlet')
    partition_parser.add_argument("--client_num", type=int, default=10)
    partition_parser.add_argument("--alpha", type=float, default=0.2)
    partition_parser.add_argument("--num_classes_per_client", type=int, default=2)
    partition_parser.add_argument("--dataset_root", type=str, default='../torchvision')
    partition_parser.add_argument("--seed", type=int, default=42)
    args = partition_parser.parse_args()
    partition(args)