# Federated Domain Generalization with Data-free On-server Gradient Matching - ICLR 2025 - Pytorch Official Implementation

## Paper Link: https://arxiv.org/abs/2501.14653

# Abstract
<p align="justify">Domain Generalization (DG) aims to learn from multiple known source domains a model that can generalize well to unknown target domains. 
One of the key approaches in DG is training an encoder which generates domain-invariant representations. However, this approach is not applicable in Federated Domain Generalization (FDG), where data from various domains are distributed across different clients. In this paper, we introduce a novel approach, dubbed Federated Learning via On-server Matching Gradient (FedOMG), which can \emph{efficiently leverage domain information from distributed domains}. Specifically, we utilize the local gradients as information about the distributed models to find an invariant gradient direction across all domains through gradient inner product maximization. The advantages are two-fold: 1) FedOMG can aggregate the characteristics of distributed models on the centralized server without incurring any additional communication cost, and 2) FedOMG is orthogonal to many existing FL/FDG methods, allowing for additional performance improvements by being seamlessly integrated with them. Extensive experimental evaluations on various settings to demonstrate the robustness of FedOMG compared to other FL/FDG baselines. Our method outperforms recent SOTA baselines on four FL benchmark datasets (MNIST, EMNIST, CIFAR-10, and CIFAR-100), and three FDG benchmark datasets (PACS, VLCS, and OfficeHome).</p>

# Experiment
## Setup
This work can be conducted on any platform: Windows, Ubuntu, Google Colab. In Windows or Ubuntu use the following script to create a virtual environment.
```
git clone https://github.com/skydvn/FedOMG.git
cd path/to/FedOMG
python -m venv .env
```

## Running

For CIFAR-10, the dataset generator stores each partition under
`Cifar10/<num_clients>/<balance>_<noniid>_<alpha>/`.

Generate the two Dirichlet partitions from the `dataset` directory if they
do not already exist:

```bash
cd dataset
python generate_cifar10.py noniid balance dir 100 0.1
python generate_cifar10.py noniid balance dir 100 0.5
cd ..
```

Run the 800-round, five-seed experiments from the repository root.

**Reproduction note:** Table 1's caption states 20% user participation, but the
FedAvg and FedOMG CIFAR-10 α=0.1 values printed in Table 1 (60.89% and 90.40%)
match exactly the 100%-participation rows in Appendix Table 5. The commands
below use `-jr 1.0` to target the paper's reported numerical values. This
follows the appendix's full-participation setting rather than the literal
Table 1 caption.

**CIFAR-10, Dirichlet α = 0.1 — FedAvg**

```bash
python system/main.py -data Cifar10 -algo FedAvg -m cnn -mstr cnn -nb 10 -nc 100 -jr 1.0 -gr 800 -ls 5 -lr 0.005 -lbs 16 -t 5 -go cifar10_a01_jr1_fedavg --noniid --balance --alpha_dirich 0.1 --seed 2025
```

**CIFAR-10, Dirichlet α = 0.1 — FedOMG**

```bash
python system/main.py -data Cifar10 -algo FedOMG -m cnn -mstr cnn -nb 10 -nc 100 -jr 1.0 -gr 800 -ls 5 -lr 0.005 -lbs 16 -t 5 -go cifar10_a01_jr1_fedomg -omgr 25 -omglr 25 -mmt 0.5 -ss 30 -gam 0.5 --omg_c 0.5 --omg_meta_lr 1.0 --noniid --balance --alpha_dirich 0.1 --seed 2025
```

**CIFAR-10, Dirichlet α = 0.5**

```bash
python system/main.py -data Cifar10 -algo FedOMG -m cnn -mstr cnn -nb 10 -nc 100 -jr 1.0 -gr 800 -ls 5 -lr 0.005 -lbs 16 -t 5 -go cifar10_a05_jr1 -omgr 25 -omglr 25 -mmt 0.5 -ss 30 -gam 0.5 --omg_c 0.5 --omg_meta_lr 1.0 --noniid --balance --alpha_dirich 0.5 --seed 2025
```

Use a distinct `-go` value for each experiment setting so the result files
for different α values are not overwritten. Add `--debug` only for short
diagnostic runs; leave it off for the full experiments.

# Citation
```
@inproceedings{
  2025-FDG-FedOMG,
  title={Federated Domain Generalization with Data-free On-server Gradient Matching},
  author={Trong-Binh Nguyen and Minh-Duong Nguyen and Jinsun Park and Quoc-Viet Pham and Won Joo Hwang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  month = {May},
  year={2025},
  url={https://openreview.net/forum?id=8TERgu1Lb2}
}
```
