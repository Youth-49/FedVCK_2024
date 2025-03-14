# FedVCK@AAAI 2025

FedVCK: Non-IID Robust and Communication-Efficient Federated Learning via Valuable Condensed Knowledge for Medical Image Analysis, Accepted by AAAI 2025



## Abstract

Federated learning has become a promising solution for collaboration among medical institutions. However, data owned by each institution would be highly heterogeneous and the distribution is always non-independent and identical distribution (non-IID), resulting in client drift and unsatisfactory performance. Despite existing federated learning methods attempting to solve the non-IID problems, they still show marginal advantages but rely on frequent communication which would incur high costs and privacy concerns. In this paper, we propose a novel federated learning method: **Fed**erated learning via **V**aluable **C**ondensed **K**nowledge (FedVCK). We enhance the quality of condensed knowledge and select the most necessary knowledge guided by models, to tackle the non-IID problem within limited communication budgets effectively. Specifically, on the client side, we condense the knowledge of each client into a small dataset and further enhance the condensation procedure with latent distribution constraints, facilitating the effective capture of high-quality knowledge. During each round, we specifically target and condense knowledge that has not been assimilated by the current model, thereby preventing unnecessary repetition of homogeneous knowledge and minimizing the frequency of communications required. On the server side, we propose relational supervised contrastive learning to provide more supervision signals to aid the global model updating. Comprehensive experiments across various medical tasks show that FedVCK can outperform state-of-the-art methods, demonstrating that it's non-IID robust and communication-efficient. 



-----



Software Environment: see `requirements.yaml`

Hardware Platform: Ubuntu with Geforce RTX 3090 GPU

Supported datasets: {Path, OCT, OrganS, OranC, Pneumuonia)MNIST, CIFAR10, STL10, ImageNette.



To run the code:

```bash
bash run.sh
```





