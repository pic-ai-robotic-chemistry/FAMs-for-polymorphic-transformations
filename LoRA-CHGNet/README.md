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
