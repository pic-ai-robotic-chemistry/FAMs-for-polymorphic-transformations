import numpy as np
from pymatgen.core import Structure
from chgnet.utils import read_json
from chgnet.data.dataset import StructureData, get_train_val_test_loader

dataset_dict = read_json("./my_vasp_calc_dir/chgnet_dataset.json")
structures = [Structure.from_dict(struct) for struct in dataset_dict["structure"]]
energies_per_atom = dataset_dict["energy_per_atom"]
forces_dict = dataset_dict["force"]
forces = []
for i in range(len(forces_dict)):
    i_image_forces = np.array(forces_dict[i]["data"])
    forces.append(i_image_forces)
stresses = None
magmoms = None

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

# Optionally fix the weights of some layers
for layer in [
    chgnet.atom_embedding,
    chgnet.bond_embedding,
    chgnet.angle_embedding,
    chgnet.bond_basis_expansion,
    chgnet.angle_basis_expansion,
]:
    for param in layer.parameters():
        param.requires_grad = False

# Define Trainer
trainer = Trainer(
    model=chgnet,
    targets="ef",
    optimizer="Adam",
    scheduler="CosLR",
    criterion="MSE",
    epochs=30,
    learning_rate=1e-2,
    use_device="cuda",
    print_freq=6,
)

trainer.train(train_loader, val_loader, test_loader, train_composition_model=True)
# model = trainer.model
# best_model = trainer.best_model  # best model based on validation energy MAE

# Out of box prediction:
# trainer._validate(train_loader, is_test=True, test_result_save_path="./out_of_box_prediction/result_traindataset")
# trainer._validate(val_loader, is_test=True, test_result_save_path="./out_of_box_prediction/result_validdataset")
# trainer._validate(test_loader, is_test=True, test_result_save_path="./out_of_box_prediction/result_testdataset")


