# CHGNet Tuning Project

## Environment Setup
Create and Activate Conda Environment
```sh
git clone git@github.com:littleyu192/CHGNet_tuning.git
conda create -n chgnet python=3.10
conda activate chgnet
cd CHGNet_tuning
pip install .
```

## Tuning Methods
1 Freeze Tuning
```sh
cd freezetuning
python default_tuning.py
```

2 Full Fine-tuning
```sh
cd fullfinetuning
python default_tuning.py
```

3 LoRA Tuning
```sh
cd loratuning
python lora_tuning.py
```

## Transition State Testing
```sh
cd ../example
python md_ramp_npt_task1_new_CHGNET.py \
    --file_path alpha_212.cif \
    --model /home/husiyu/Project/Finetuning/CHGNet/lora_tuned_models_v1/lora_v2_pth/epoch9_e4_f66_sNA_mNA \
    --device "cuda:0" \
    --xyz_filename "alpha_212_CHGNet_lora_tuned_v1_pt.xyz" \
    --log_filename "alpha_212_CHGNet_lora_tuned_v1_pt.log" \
    --traj_filename "alpha_212_CHGNet_lora_tuned_v1_pt.traj" \
    --mode increase
```
