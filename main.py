import copy
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Subset

from src.client import Client
from src.server import Server
from config import parser
from dataset.data.dataset import get_dataset, PerLabelDatasetNonIID
from src.utils import setup_seed, get_model, ParamDiffAug
import logging

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy == 'None' else True

    
    if args.partition == 'dirichlet':
        split_file = f'/{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
        args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)
        if args.compression_ratio > 0.:
            model_identification = f'{args.dataset}_alpha{args.alpha}_{args.client_num}clients/{args.model}_{100*args.compression_ratio}%_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        else:
            model_identification = f'{args.dataset}_alpha{args.alpha}_{args.client_num}clients/{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
            # raise Exception('Compression ratio should > 0')
    elif args.partition == 'label':
        split_file = f'/{args.dataset}_client_num={args.client_num}_label={args.num_classes_per_client}.json'
        args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)
        if args.compression_ratio > 0.:
            model_identification = f'{args.dataset}_label{args.num_classes_per_client}_{args.client_num}clients/{args.model}_{100*args.compression_ratio}%_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        else:
            raise Exception('Compression ratio should > 0')
    elif args.partition == 'pathological':
        split_file = f'/{args.dataset}_client_num={args.client_num}_pathological={args.num_classes_per_client}.json'
        args.split_file = os.path.join(os.path.dirname(__file__), "dataset/split_file"+split_file)
        if args.compression_ratio > 0.:
            model_identification = f'{args.dataset}_pathological{args.num_classes_per_client}_{args.client_num}clients/{args.model}_{100*args.compression_ratio}%_{args.dc_iterations}dc_{args.model_epochs}epochs_{args.tag}'
        else:
            raise Exception('Compression ratio should > 0')

    args.save_root_path = os.path.join(os.path.dirname(__file__), 'results/')
    args.save_root_path = os.path.join(args.save_root_path, model_identification)
    os.makedirs(args.save_root_path, exist_ok=True)
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    log_file = 'log.txt'
    log_path = os.path.join(args.save_root_path, log_file)
    print(log_path)
    if os.path.exists(log_path):
        raise Exception('log file already exists!')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    setup_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # get dataset and init models
    dataset_info, train_set, test_set, test_loader = get_dataset(args.dataset, args.dataset_root, args.batch_size)
    print("load data: done")
    with open(args.split_file, 'r') as file:
        file_data = json.load(file)
    client_indices, client_classes = file_data['client_idx'], file_data['client_classes']

    if args.dataset in ['CIFAR10', 'FMNIST',]:
        labels = np.array(train_set.targets, dtype='int64')
    elif args.dataset in ['PathMNIST', 'OCTMNIST', 'OrganSMNIST', 'OrganCMNIST', 'ImageNette', 'OrganCMNIST224', 'PneumoniaMNIST224', 'RetinaMNIST224', 'STL', 'STL32']:
        labels = train_set.labels
    net_cls_counts = {}
    dict_users = {i: idcs for i, idcs in enumerate(client_indices)}
    for net_i, dataidx in dict_users.items():
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logging.info(f'Data statistics: {net_cls_counts}')
    logging.info(f'client classes: {client_classes}')
    
    train_sets = [Subset(train_set, indices) for indices in client_indices]

    global_model = get_model(args.model, dataset_info)
    logging.info(global_model)
    logging.info(get_n_params(global_model))
    logging.info(args.__dict__)

    # init server and clients
    client_list = [Client(
        cid=i,
        train_set=PerLabelDatasetNonIID(
            train_sets[i],
            client_classes[i],
            dataset_info['channel'],
            device,
        ),
        classes=client_classes[i],
        dataset_info=dataset_info,
        ipc=args.ipc,
        compression_ratio=args.compression_ratio,
        dc_iterations=args.dc_iterations,
        real_batch_size=args.dc_batch_size,
        image_lr=args.image_lr,
        image_momentum=args.image_momentum,
        image_weight_decay=args.image_weight_decay,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        local_ep=args.local_ep,
        dsa=args.dsa,
        dsa_strategy=args.dsa_strategy,
        init = args.init,
        clip_norm = args.clip_norm,
        gamma = args.gamma,
        lamda = args.lamda,
        b = args.b,
        con_temp = args.con_temp,
        kernel = args.kernel,
        save_root_path=args.save_root_path,
        device=device,
    ) for i in range(args.client_num)]

    server = Server(
        train_set = PerLabelDatasetNonIID(
            train_set,
            range(0,dataset_info['num_classes']),
            dataset_info['channel'],
            device,
        ),
        ipc = args.ipc,
        dataset_info=dataset_info,
        global_model_name=args.model,
        global_model=global_model,
        clients=client_list,
        communication_rounds=args.communication_rounds,
        join_ratio=args.join_ratio,
        batch_size=args.batch_size,
        model_epochs=args.model_epochs,
        lr_server=args.lr_server,
        momentum_server=args.momentum_server,
        weight_decay_server=args.weight_decay_server,
        lr_head=args.lr_head,
        momentum_head=args.momentum_head, 
        weight_decay_head=args.weight_decay_head,
        weighted_matching = args.weighted_matching,
        weighted_sample = args.weighted_sample,
        weighted_mmd = args.weighted_mmd,
        contrastive_way = args.contrastive_way,
        con_beta = args.con_beta,
        con_temp = args.con_temp,
        topk = args.topk,
        dsa = args.dsa,
        dsa_strategy = args.dsa_strategy,
        preserve_all = args.preserve_all,
        eval_gap=args.eval_gap,
        test_set=test_set,
        test_loader=test_loader,
        device=device,
        model_identification=model_identification,
        save_root_path=args.save_root_path
    )
    print('Server and Clients have been created.')

    # fit the model
    server.fit()

if __name__ == "__main__":
    main()