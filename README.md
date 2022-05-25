# Codes for [Analyzing Lottery Ticket Hypothesis from PAC-Bayesian Theory Perspective](https://arxiv.org/abs/2205.07320)

To evaluate the PAC-Bayes bound in the LTH setting, we used the modified version of [open_lth](https://github.com/facebookresearch/open_lth) and [role-of-data](https://github.com/kylehkhsu/role-of-data).

## Install
```bash
git clone https://github.com/keitaroskmt/lth-pacbayes.git
cd lth-pacbayes
```

## Run codes
Rum IMP and create results under `lottery_data`. The following command corresponds to the ResNet20 setting of LTH original paper.
```bash
python main.py --lr=0.01 --model_name=resnet20 --dataset_name=CIFAR10 --iteration=30000 --milestone_steps=20000,25000 --weight_decay=1e-4 --momentum=0.9 --pruning_layers_to_ignore=fc.weight --gpu=0
```

Evaluate the PAC-Bayes bound for the target subnetwork. You should specify the hash name under `lottery_data`.
```bash
python bound/main.py --hash_name=ff043db27567f3d00ed1f8ed7c9c7e0e --dist_type=spike-and-slab
```