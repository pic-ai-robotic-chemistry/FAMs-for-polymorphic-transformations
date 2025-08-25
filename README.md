# FAMs-for-polymorphic-transformations
This repository contains the implementation of our ELoRA for Equivariant GNNs and LoRA for Invariant GNN:

The code is built upon the e3nn library for the MACE and SevenNet framework and peft library for CHGNet framework.

## ELoRA for fine-tune foundation models of MACE 
```bash
mace_run_train \
    --name="MACE_model" \
    --train_file="./dataset/train.xyz" \
    --valid_fraction=0.1 \
    --test_file="./dataset/test.xyz" \
    --E0s='average' \
    --foundation_model="MACE-OFF23_medium.model" \
    --model="MACE" \
    --loss="ef" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --r_max=5.0 \
    --lr=0.005 \
    --forces_weight=1000 \
    --energy_weight=1 \
    --weight_decay=1e-8 \
    --clip_grad=100 \
    --batch_size=5 \
    --valid_batch_size=5 \
    --max_num_epochs=500 \
    --scheduler_patience=5 \
    --ema \
    --ema_decay=0.995 \
    --error_table="TotalRMSE" \
    --default_dtype="float64"\
    --device=cuda \
    --seed=123 \
    --save_cpu 
```
## ELoRA for fine-tune foundation models of SevenNet 
```bash
python train_sevenn.py \
       --pretrained 7net-0 \
       --xyz xyz_files/atoms_7774.xyz \
       --device cuda:0 \
       --energy-weight 1.0 \
       --force-weight 25.0 \
       --stress-weight 0.01 \
       --epochs 100 \
       --batch-size 20 \
       --lr 0.004 \
       --train-rescale \
       --out ./checkpoints/Elora_sevennet.pth
```



You may adjust the hyperparameters or input files to suit your specific dataset or evaluation setting.

