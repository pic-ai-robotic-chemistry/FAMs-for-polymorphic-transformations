import numpy as np
from pymatgen.core import Structure
from chgnet.utils import read_json
from chgnet.data.dataset import StructureData, get_train_val_test_loader

dataset_dict = read_json("../my_vasp_calc_dir/chgnet_dataset.json")
structures = [Structure.from_dict(struct) for struct in dataset_dict["structure"]]
energies_per_atom = dataset_dict["energy_per_atom"]
forces_dict = dataset_dict["force"]
forces = []
for i in range(len(forces_dict)):
    i_image_forces = np.array(forces_dict[i]["data"])
    forces.append(i_image_forces)
stresses = None
magmoms = None

# import ipdb; ipdb.set_trace()

dataset = StructureData(
    structures=structures,
    energies=energies_per_atom,
    forces=forces,
    stresses=stresses,  # can be None
    magmoms=magmoms,  # can be None
)
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset, batch_size=8, train_ratio=0.85, val_ratio=0.05
)

from chgnet.model import CHGNet
from chgnet.trainer import Trainer

# Load pretrained CHGNet
chgnet = CHGNet.load()

from peft import LoraConfig, get_peft_model

# composition_model
# composition_model.fc
# graph_converter
# atom_embedding
# atom_embedding.embedding
# bond_basis_expansion
# bond_basis_expansion.rbf_expansion_ag
# bond_basis_expansion.rbf_expansion_ag.smooth_cutoff
# bond_basis_expansion.rbf_expansion_bg
# bond_basis_expansion.rbf_expansion_bg.smooth_cutoff
# bond_embedding
# bond_weights_ag
# bond_weights_bg
# angle_basis_expansion
# angle_basis_expansion.fourier_expansion
# angle_embedding
# atom_conv_layers
# atom_conv_layers.0
# atom_conv_layers.0.activation
# atom_conv_layers.0.twoBody_atom
# atom_conv_layers.0.twoBody_atom.mlp_core
# atom_conv_layers.0.twoBody_atom.mlp_core.layers
# atom_conv_layers.0.twoBody_atom.mlp_core.layers.0
# atom_conv_layers.0.twoBody_atom.mlp_core.layers.1
# atom_conv_layers.0.twoBody_atom.mlp_core.layers.2
# atom_conv_layers.0.twoBody_atom.mlp_core.layers.3
# atom_conv_layers.0.twoBody_atom.mlp_gate
# atom_conv_layers.0.twoBody_atom.mlp_gate.layers
# atom_conv_layers.0.twoBody_atom.mlp_gate.layers.0
# atom_conv_layers.0.twoBody_atom.mlp_gate.layers.1
# atom_conv_layers.0.twoBody_atom.mlp_gate.layers.2
# atom_conv_layers.0.twoBody_atom.mlp_gate.layers.3
# atom_conv_layers.0.twoBody_atom.activation
# atom_conv_layers.0.twoBody_atom.sigmoid
# atom_conv_layers.0.twoBody_atom.bn1
# atom_conv_layers.0.twoBody_atom.bn2
# atom_conv_layers.0.mlp_out
# atom_conv_layers.0.mlp_out.layers
# atom_conv_layers.0.mlp_out.layers.0
# atom_conv_layers.0.mlp_out.layers.1
# atom_conv_layers.1
# atom_conv_layers.1.activation
# atom_conv_layers.1.twoBody_atom
# atom_conv_layers.1.twoBody_atom.mlp_core
# atom_conv_layers.1.twoBody_atom.mlp_core.layers
# atom_conv_layers.1.twoBody_atom.mlp_core.layers.0
# atom_conv_layers.1.twoBody_atom.mlp_core.layers.1
# atom_conv_layers.1.twoBody_atom.mlp_core.layers.2
# atom_conv_layers.1.twoBody_atom.mlp_core.layers.3
# atom_conv_layers.1.twoBody_atom.mlp_gate
# atom_conv_layers.1.twoBody_atom.mlp_gate.layers
# atom_conv_layers.1.twoBody_atom.mlp_gate.layers.0
# atom_conv_layers.1.twoBody_atom.mlp_gate.layers.1
# atom_conv_layers.1.twoBody_atom.mlp_gate.layers.2
# atom_conv_layers.1.twoBody_atom.mlp_gate.layers.3
# atom_conv_layers.1.twoBody_atom.activation
# atom_conv_layers.1.twoBody_atom.sigmoid
# atom_conv_layers.1.twoBody_atom.bn1
# atom_conv_layers.1.twoBody_atom.bn2
# atom_conv_layers.1.mlp_out
# atom_conv_layers.1.mlp_out.layers
# atom_conv_layers.1.mlp_out.layers.0
# atom_conv_layers.1.mlp_out.layers.1
# atom_conv_layers.2
# atom_conv_layers.2.activation
# atom_conv_layers.2.twoBody_atom
# atom_conv_layers.2.twoBody_atom.mlp_core
# atom_conv_layers.2.twoBody_atom.mlp_core.layers
# atom_conv_layers.2.twoBody_atom.mlp_core.layers.0
# atom_conv_layers.2.twoBody_atom.mlp_core.layers.1
# atom_conv_layers.2.twoBody_atom.mlp_core.layers.2
# atom_conv_layers.2.twoBody_atom.mlp_core.layers.3
# atom_conv_layers.2.twoBody_atom.mlp_gate
# atom_conv_layers.2.twoBody_atom.mlp_gate.layers
# atom_conv_layers.2.twoBody_atom.mlp_gate.layers.0
# atom_conv_layers.2.twoBody_atom.mlp_gate.layers.1
# atom_conv_layers.2.twoBody_atom.mlp_gate.layers.2
# atom_conv_layers.2.twoBody_atom.mlp_gate.layers.3
# atom_conv_layers.2.twoBody_atom.activation
# atom_conv_layers.2.twoBody_atom.sigmoid
# atom_conv_layers.2.twoBody_atom.bn1
# atom_conv_layers.2.twoBody_atom.bn2
# atom_conv_layers.2.mlp_out
# atom_conv_layers.2.mlp_out.layers
# atom_conv_layers.2.mlp_out.layers.0
# atom_conv_layers.2.mlp_out.layers.1
# atom_conv_layers.3
# atom_conv_layers.3.activation
# atom_conv_layers.3.twoBody_atom
# atom_conv_layers.3.twoBody_atom.mlp_core
# atom_conv_layers.3.twoBody_atom.mlp_core.layers
# atom_conv_layers.3.twoBody_atom.mlp_core.layers.0
# atom_conv_layers.3.twoBody_atom.mlp_core.layers.1
# atom_conv_layers.3.twoBody_atom.mlp_core.layers.2
# atom_conv_layers.3.twoBody_atom.mlp_core.layers.3
# atom_conv_layers.3.twoBody_atom.mlp_gate
# atom_conv_layers.3.twoBody_atom.mlp_gate.layers
# atom_conv_layers.3.twoBody_atom.mlp_gate.layers.0
# atom_conv_layers.3.twoBody_atom.mlp_gate.layers.1
# atom_conv_layers.3.twoBody_atom.mlp_gate.layers.2
# atom_conv_layers.3.twoBody_atom.mlp_gate.layers.3
# atom_conv_layers.3.twoBody_atom.activation
# atom_conv_layers.3.twoBody_atom.sigmoid
# atom_conv_layers.3.twoBody_atom.bn1
# atom_conv_layers.3.twoBody_atom.bn2
# atom_conv_layers.3.mlp_out
# atom_conv_layers.3.mlp_out.layers
# atom_conv_layers.3.mlp_out.layers.0
# atom_conv_layers.3.mlp_out.layers.1
# bond_conv_layers
# bond_conv_layers.0
# bond_conv_layers.0.activation
# bond_conv_layers.0.twoBody_bond
# bond_conv_layers.0.twoBody_bond.mlp_core
# bond_conv_layers.0.twoBody_bond.mlp_core.layers
# bond_conv_layers.0.twoBody_bond.mlp_core.layers.0
# bond_conv_layers.0.twoBody_bond.mlp_core.layers.1
# bond_conv_layers.0.twoBody_bond.mlp_core.layers.2
# bond_conv_layers.0.twoBody_bond.mlp_core.layers.3
# bond_conv_layers.0.twoBody_bond.mlp_gate
# bond_conv_layers.0.twoBody_bond.mlp_gate.layers
# bond_conv_layers.0.twoBody_bond.mlp_gate.layers.0
# bond_conv_layers.0.twoBody_bond.mlp_gate.layers.1
# bond_conv_layers.0.twoBody_bond.mlp_gate.layers.2
# bond_conv_layers.0.twoBody_bond.mlp_gate.layers.3
# bond_conv_layers.0.twoBody_bond.activation
# bond_conv_layers.0.twoBody_bond.sigmoid
# bond_conv_layers.0.twoBody_bond.bn1
# bond_conv_layers.0.twoBody_bond.bn2
# bond_conv_layers.0.mlp_out
# bond_conv_layers.0.mlp_out.layers
# bond_conv_layers.0.mlp_out.layers.0
# bond_conv_layers.0.mlp_out.layers.1
# bond_conv_layers.1
# bond_conv_layers.1.activation
# bond_conv_layers.1.twoBody_bond
# bond_conv_layers.1.twoBody_bond.mlp_core
# bond_conv_layers.1.twoBody_bond.mlp_core.layers
# bond_conv_layers.1.twoBody_bond.mlp_core.layers.0
# bond_conv_layers.1.twoBody_bond.mlp_core.layers.1
# bond_conv_layers.1.twoBody_bond.mlp_core.layers.2
# bond_conv_layers.1.twoBody_bond.mlp_core.layers.3
# bond_conv_layers.1.twoBody_bond.mlp_gate
# bond_conv_layers.1.twoBody_bond.mlp_gate.layers
# bond_conv_layers.1.twoBody_bond.mlp_gate.layers.0
# bond_conv_layers.1.twoBody_bond.mlp_gate.layers.1
# bond_conv_layers.1.twoBody_bond.mlp_gate.layers.2
# bond_conv_layers.1.twoBody_bond.mlp_gate.layers.3
# bond_conv_layers.1.twoBody_bond.activation
# bond_conv_layers.1.twoBody_bond.sigmoid
# bond_conv_layers.1.twoBody_bond.bn1
# bond_conv_layers.1.twoBody_bond.bn2
# bond_conv_layers.1.mlp_out
# bond_conv_layers.1.mlp_out.layers
# bond_conv_layers.1.mlp_out.layers.0
# bond_conv_layers.1.mlp_out.layers.1
# bond_conv_layers.2
# bond_conv_layers.2.activation
# bond_conv_layers.2.twoBody_bond
# bond_conv_layers.2.twoBody_bond.mlp_core
# bond_conv_layers.2.twoBody_bond.mlp_core.layers
# bond_conv_layers.2.twoBody_bond.mlp_core.layers.0
# bond_conv_layers.2.twoBody_bond.mlp_core.layers.1
# bond_conv_layers.2.twoBody_bond.mlp_core.layers.2
# bond_conv_layers.2.twoBody_bond.mlp_core.layers.3
# bond_conv_layers.2.twoBody_bond.mlp_gate
# bond_conv_layers.2.twoBody_bond.mlp_gate.layers
# bond_conv_layers.2.twoBody_bond.mlp_gate.layers.0
# bond_conv_layers.2.twoBody_bond.mlp_gate.layers.1
# bond_conv_layers.2.twoBody_bond.mlp_gate.layers.2
# bond_conv_layers.2.twoBody_bond.mlp_gate.layers.3
# bond_conv_layers.2.twoBody_bond.activation
# bond_conv_layers.2.twoBody_bond.sigmoid
# bond_conv_layers.2.twoBody_bond.bn1
# bond_conv_layers.2.twoBody_bond.bn2
# bond_conv_layers.2.mlp_out
# bond_conv_layers.2.mlp_out.layers
# bond_conv_layers.2.mlp_out.layers.0
# bond_conv_layers.2.mlp_out.layers.1
# angle_layers
# angle_layers.0
# angle_layers.0.activation
# angle_layers.0.twoBody_bond
# angle_layers.0.twoBody_bond.mlp_core
# angle_layers.0.twoBody_bond.mlp_core.layers
# angle_layers.0.twoBody_bond.mlp_core.layers.0
# angle_layers.0.twoBody_bond.mlp_core.layers.1
# angle_layers.0.twoBody_bond.mlp_gate
# angle_layers.0.twoBody_bond.mlp_gate.layers
# angle_layers.0.twoBody_bond.mlp_gate.layers.0
# angle_layers.0.twoBody_bond.mlp_gate.layers.1
# angle_layers.0.twoBody_bond.activation
# angle_layers.0.twoBody_bond.sigmoid
# angle_layers.0.twoBody_bond.bn1
# angle_layers.0.twoBody_bond.bn2
# angle_layers.1
# angle_layers.1.activation
# angle_layers.1.twoBody_bond
# angle_layers.1.twoBody_bond.mlp_core
# angle_layers.1.twoBody_bond.mlp_core.layers
# angle_layers.1.twoBody_bond.mlp_core.layers.0
# angle_layers.1.twoBody_bond.mlp_core.layers.1
# angle_layers.1.twoBody_bond.mlp_gate
# angle_layers.1.twoBody_bond.mlp_gate.layers
# angle_layers.1.twoBody_bond.mlp_gate.layers.0
# angle_layers.1.twoBody_bond.mlp_gate.layers.1
# angle_layers.1.twoBody_bond.activation
# angle_layers.1.twoBody_bond.sigmoid
# angle_layers.1.twoBody_bond.bn1
# angle_layers.1.twoBody_bond.bn2
# angle_layers.2
# angle_layers.2.activation
# angle_layers.2.twoBody_bond
# angle_layers.2.twoBody_bond.mlp_core
# angle_layers.2.twoBody_bond.mlp_core.layers
# angle_layers.2.twoBody_bond.mlp_core.layers.0
# angle_layers.2.twoBody_bond.mlp_core.layers.1
# angle_layers.2.twoBody_bond.mlp_gate
# angle_layers.2.twoBody_bond.mlp_gate.layers
# angle_layers.2.twoBody_bond.mlp_gate.layers.0
# angle_layers.2.twoBody_bond.mlp_gate.layers.1
# angle_layers.2.twoBody_bond.activation
# angle_layers.2.twoBody_bond.sigmoid
# angle_layers.2.twoBody_bond.bn1
# angle_layers.2.twoBody_bond.bn2
# site_wise
# readout_norm
# pooling
# mlp
# mlp.layers
# mlp.layers.0
# mlp.layers.1
# mlp.layers.2
# mlp.layers.3
# mlp.layers.4
# mlp.layers.5
# mlp.layers.6
# mlp.layers.7

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=16,  # 缩放因子
    target_modules=["composition_model.fc",
                    "bond_embedding",                
                    "bond_weights_ag",
                    "bond_weights_bg",
                    "angle_embedding",
                    "atom_conv_layers.0.twoBody_atom.mlp_core.layers.0",
                    "atom_conv_layers.0.twoBody_atom.mlp_core.layers.3",
                    "atom_conv_layers.0.twoBody_atom.mlp_gate.layers.0",
                    "atom_conv_layers.0.twoBody_atom.mlp_gate.layers.3",
                    "atom_conv_layers.0.mlp_out.layers.1",
                    "atom_conv_layers.1.twoBody_atom.mlp_core.layers.0",
                    "atom_conv_layers.1.twoBody_atom.mlp_core.layers.3",
                    "atom_conv_layers.1.twoBody_atom.mlp_gate.layers.0",
                    "atom_conv_layers.1.twoBody_atom.mlp_gate.layers.3",
                    "atom_conv_layers.1.mlp_out.layers.1",
                    "atom_conv_layers.2.twoBody_atom.mlp_core.layers.0",
                    "atom_conv_layers.2.twoBody_atom.mlp_core.layers.3",
                    "atom_conv_layers.2.twoBody_atom.mlp_gate.layers.0",
                    "atom_conv_layers.2.twoBody_atom.mlp_gate.layers.3",
                    "atom_conv_layers.2.mlp_out.layers.1",
                    "atom_conv_layers.3.twoBody_atom.mlp_core.layers.0",
                    "atom_conv_layers.3.twoBody_atom.mlp_core.layers.3",
                    "atom_conv_layers.3.twoBody_atom.mlp_gate.layers.0",
                    "atom_conv_layers.3.twoBody_atom.mlp_gate.layers.3",
                    "atom_conv_layers.3.mlp_out.layers.1",
                    "bond_conv_layers.0.twoBody_bond.mlp_core.layers.0",
                    "bond_conv_layers.0.twoBody_bond.mlp_core.layers.3",
                    "bond_conv_layers.0.twoBody_bond.mlp_gate.layers.0",
                    "bond_conv_layers.0.twoBody_bond.mlp_gate.layers.3",
                    "bond_conv_layers.0.mlp_out.layers.1",
                    "bond_conv_layers.1.twoBody_bond.mlp_core.layers.0",
                    "bond_conv_layers.1.twoBody_bond.mlp_core.layers.3",
                    "bond_conv_layers.1.twoBody_bond.mlp_gate.layers.0",
                    "bond_conv_layers.1.twoBody_bond.mlp_gate.layers.3",
                    "bond_conv_layers.1.mlp_out.layers.1",
                    "bond_conv_layers.2.twoBody_bond.mlp_core.layers.0",
                    "bond_conv_layers.2.twoBody_bond.mlp_core.layers.3",
                    "bond_conv_layers.2.twoBody_bond.mlp_gate.layers.0",
                    "bond_conv_layers.2.twoBody_bond.mlp_gate.layers.3",
                    "bond_conv_layers.2.mlp_out.layers.1",
                    "angle_layers.0.twoBody_bond.mlp_core.layers.1",
                    "angle_layers.0.twoBody_bond.mlp_gate.layers.1",
                    "angle_layers.1.twoBody_bond.mlp_core.layers.1",
                    "angle_layers.1.twoBody_bond.mlp_gate.layers.1",
                    "angle_layers.2.twoBody_bond.mlp_core.layers.1",
                    "angle_layers.2.twoBody_bond.mlp_gate.layers.1",
                    "site_wise",
                    "mlp.layers.0",
                    "mlp.layers.2",
                    "mlp.layers.4",
                    "mlp.layers.7"],  # 应用 LoRA 的模块
    lora_dropout=0.1,
    bias="none",
)

for name, module in chgnet.named_modules():
    print(name, module)
# import ipdb; ipdb.set_trace()



# 将 LoRA 应用到模型
chgnet = get_peft_model(chgnet, lora_config)
chgnet.save_pretrained("lora_model")

# Optionally fix the weights of some layers
# for layer in [
#     chgnet.atom_embedding,
#     chgnet.bond_embedding,
#     chgnet.angle_embedding,
#     chgnet.bond_basis_expansion,
#     chgnet.angle_basis_expansion,
# ]:
#     for param in layer.parameters():
#         param.requires_grad = False

# Define Trainer
trainer = Trainer(
    model=chgnet,
    targets="ef",
    optimizer="Adam",
    scheduler="CosLR",
    criterion="MSE",
    epochs=1,
    learning_rate=1e-2,
    use_device="cuda",
    print_freq=6,
    is_lora=True,
)

trainer.train(train_loader, val_loader, test_loader, train_composition_model=True)

# model = trainer.model
# best_model = trainer.best_model  # best model based on validation energy MAE


