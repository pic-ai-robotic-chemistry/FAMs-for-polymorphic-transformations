import random
from datetime import datetime
from ase import units
from ase.io import write
from ase.md.npt import NPT
# from chgnet.model.model_change import CHGNet # 这里吧model.py 文件修改了，加了一个函数，用于加载模型
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.io.trajectory import Trajectory, TrajectoryReader
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import argparse
import sys

# 运行命令：
# python md_ramp_npt_task1_new_CHGNET.py --file_path alpha_212.cif --model ./lora_tuned_models/lora_tuned_v1_pth/bestE_epoch9_e6_f168_sNA_mNA --device "cuda:0" --xyz_filename "alpha_212_CHGNet_lora_tuned_v1_pt.xyz" --log_filename  "alpha_212_CHGNet_lora_tuned_v1_pt.log" --traj_filename  "alpha_212_CHGNet_lora_tuned_v1_pt.traj" --mode increase


def seed_everything(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(
        description="Run NPT MD simulation with custom pressure ramp and logging intervals.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CIF file.')
    parser.add_argument('--model', type=str, required=True, help='Path or name of the MACE model.')
    parser.add_argument('--device', type=str, required=True, help='Compute device, e.g. "cpu" or "cuda".')
    parser.add_argument('--xyz_filename', type=str, default='simulation_npt.xyz', help='Output XYZ filename.')
    parser.add_argument('--log_filename', type=str, default='simulation_npt.log', help='Log filename.')
    parser.add_argument('--traj_filename', type=str, default='simulation_npt.traj', help='traj filename.')
    parser.add_argument('--mode', type=str, choices=['increase', 'decrease'], required=True,
                        help='Simulation mode: "increase" or "decrease".')

    args = parser.parse_args()

    file_path = args.file_path
    model = args.model
    cuda = args.device
    xyz_filename = args.xyz_filename
    log_filename = args.log_filename
    mode = args.mode.lower()

    # 读取CIF文件
    structure = Structure.from_file(file_path)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    print("原始晶胞向量:")
    print(atoms.cell)

    # 确保晶胞为上三角矩阵形式
    cell = atoms.cell
    upper_triangular_cell = np.triu(cell)
    atoms.set_cell(upper_triangular_cell, scale_atoms=False)
    print("修改后的晶胞向量:")
    print(atoms.cell)
    
    from peft import PeftModel
    # CHGNet_model = CHGNet.load(model_name=model) # 都是用的0.3.0版本的模型
    chgnet = CHGNet.load()
    CHGNet_model = PeftModel.from_pretrained(chgnet, model)  #"./lora_tuned_models/lora_tuned_v1_pt"
    calc = CHGNetCalculator(model=CHGNet_model, return_site_energies=True)
    # calc = mace_off(model_path=model, device=cuda, default_dtype="float64")
    atoms.calc = calc

    # 模拟总时间：1 ns = 1,000,000 fs
    total_time = 5e5 * units.fs
    timestep = 1.0 * units.fs
    Nsteps = int(total_time / timestep)
    print("total_time:", total_time)
    print("timestep:", timestep)
    print("Nsteps:", Nsteps)
    T = 300.0  # 温度K

    MaxwellBoltzmannDistribution(atoms, T * units.kB)

    # 压力设定：根据模式从1 atm到2 GPa或从2 GPa到1 atm
    if mode == 'increase':
        initial_stress = 0.000101325
        final_stress = 0.01248
    elif mode == 'decrease':
        initial_stress = 0.01248
        final_stress = 0.000101325
    else:
        print("Invalid mode selected.")
        sys.exit(1)

    # ttime与ptime可根据需要保持不变，这里沿用之前的设置
    # 升压/降压2ns，稳定1ns
    # 假设使用之前的t与p时间参数（也可自行调整）
    taup = 1000 * timestep
    ptime = taup
    ttime = 100.0 * timestep

    B = 0.3  # 压缩模量 ev/Angstrom^3
    # 计算压力因子
    pfactor = ptime ** 2 * B * units.eV / units.Angstrom ** 3

    mass = atoms.get_masses().sum()
    print("the total mass: {}".format(mass))

    trajectory_filename = args.traj_filename

    dyn = NPT(
        atoms,
        timestep=timestep,
        temperature_K=T,
        externalstress=initial_stress,
        ttime=ttime,
        pfactor=pfactor,
        mask=None)

    traj = Trajectory(trajectory_filename, 'w', atoms)
    dyn.attach(traj.write, interval=1)

    # 定义压力变化函数：
    # 前2 ns (0 ~ 2,000,000 steps)线性变化，从initial_stress到final_stress
    # 后1 ns (2,000,000 ~ 3,000,000 steps)保持final_stress
    ramp_steps = 100000

    def get_stress_target(step):
        if step <= ramp_steps:
            return initial_stress + (final_stress - initial_stress) * (step / float(ramp_steps))
        else:
            return final_stress

    # Lattice = atoms.get_cell()
    # 输出初始XYZ
    # comment_line = f"Lattice=\n{np.array2string(Lattice, separator=', ')}"
    # write(xyz_filename, atoms, format='xyz', append=False, comment=comment_line)

    # 输出log文件
    # 初始写入表头，第一帧

    with open(log_filename, "w") as log_file:
        log_file.write(
            "# step real_time(s) sim_time(fs) epot(J) ekin(J) Temp(K) Etot(J) density(kg/m^3) a_length b_length c_length volume(m^3) pressure(Pa) pressure(GPa) pxx(Pa) pyy(Pa) pzz(Pa) pyz(Pa) pxz(Pa) pxy(Pa) target_p(GPa)\n"
        )

    start_wall_time = datetime.now()

    # 前5000 fs（5000步）每2步输出一次
    # 之后每10fs（10步）输出一次
    initial_high_freq_steps = 5000
    high_freq_interval = 1
    low_freq_interval = 1

    # 为减少开销，在后期采用批量运行：如一次dyn.run(10)后输出一次数据
    # 当步数<5000时，每步dyn.run(1)，因为需要更高频输出（每2步输出一次）
    # 当步数>5000时，可使用dyn.run(2)等批量方式，在每批结束后输出

    eV_to_J = 1.6021765e-19
    A3_to_m3 = 1e-30
    u_to_kg = 1.66053906660e-27
    mass_kg = mass * u_to_kg
    eV_A3_to_GPa = 160.21766208
    current_step = 0
    current_pressure_gpa = 0

    while current_step < Nsteps:
        if current_step < initial_high_freq_steps:
            # 前5000步，高频输出，每2步输出一次
            steps_to_run = 1  # 每次只跑1步，方便精确控制输出频率
            dyn.run(steps_to_run)
            current_step += steps_to_run

            # 判断是否输出
            if current_step % high_freq_interval == 0:
                # 输出log和xyz
                epot = atoms.get_potential_energy()
                ekin = atoms.get_kinetic_energy()
                N = len(atoms)
                temperature = 2.0 * ekin / (3.0 * units.kB * N)
                volume = atoms.get_volume()
                volume_m3 = volume * A3_to_m3
                cell = atoms.get_cell()
                a_length, b_length, c_length = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
                # print("the lengths:", a_length, b_length, c_length)
                density = mass_kg / volume_m3
                # 获取晶胞a, b, c轴长度
                a_length, b_length, c_length, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

                stress = atoms.get_stress()
                stress_pa = np.array(stress) * 1.6021765e11
                pressure_pa = -(stress_pa[0] + stress_pa[1] + stress_pa[2]) / 3.0
                pressure_gpa = pressure_pa * 1e-9

                sim_time_fs = current_step * timestep / (1.0 * units.fs)
                real_time = datetime.now() - start_wall_time

                epot_j = epot * eV_to_J
                ekin_j = ekin * eV_to_J
                etot_j = epot_j + ekin_j

                print_stat = (f"{current_step:10d} {real_time.total_seconds():.3f} {sim_time_fs:13.2f} "
                              f"{epot_j:.3e} {ekin_j:.3e} {temperature:.3f} {etot_j:.3e} "
                              f"{density:.3f} {volume_m3:.3e} {a_length:.3f} {b_length:.3f} {c_length:.3f} "
                              f"{alpha:.3f} {beta:.3f} {gamma:.3f} "  # 新增的角度输出
                              f"{pressure_pa:.3e} {pressure_gpa:.3e} "
                              f"{stress_pa[0]:.3e} {stress_pa[1]:.3e} {stress_pa[2]:.3e} "
                              f"{stress_pa[3]:.3e} {stress_pa[4]:.3e} {stress_pa[5]:.3e} "
                              f"{current_pressure_gpa:.3f}")
                comment_line = f"Lattice parameters: a={a_length:.3f}, b={b_length:.3f}, c={c_length:.3f}, alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}"

                with open(log_filename, "a") as log_file:
                    log_file.write(print_stat + "\n")
                # write(xyz_filename, atoms, format='xyz', append=True, comment=comment_line)

            # 更新应力
            current_stress = get_stress_target(current_step)
            current_pressure_gpa = current_stress * eV_A3_to_GPa

            dyn.set_stress(current_stress)

        else:
            # 后续步数，降低输出频率：每1步输出一次
            steps_to_run = low_freq_interval
            if current_step + steps_to_run > Nsteps:
                steps_to_run = Nsteps - current_step  # 确保不超出总步数

            dyn.run(steps_to_run)
            current_step += steps_to_run

            # 每1步输出一次
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            N = len(atoms)
            temperature = 2.0 * ekin / (3.0 * units.kB * N)
            volume = atoms.get_volume()
            volume_m3 = volume * A3_to_m3
            density = mass_kg / volume_m3

            a_length, b_length, c_length, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

            stress = atoms.get_stress()
            stress_pa = np.array(stress) * 1.6021765e11
            pressure_pa = -(stress_pa[0] + stress_pa[1] + stress_pa[2]) / 3.0
            pressure_gpa = pressure_pa * 1e-9

            sim_time_fs = current_step * timestep / (1.0 * units.fs)
            real_time = datetime.now() - start_wall_time

            epot_j = epot * eV_to_J
            ekin_j = ekin * eV_to_J
            etot_j = epot_j + ekin_j

            print_stat = (f"{current_step:10d} {real_time.total_seconds():.3f} {sim_time_fs:13.2f} "
                          f"{epot_j:.3e} {ekin_j:.3e} {temperature:.3f} {etot_j:.3e} "
                          f"{density:.3f} {volume_m3:.3e} {a_length:.3f} {b_length:.3f} {c_length:.3f} "
                          f"{alpha:.3f} {beta:.3f} {gamma:.3f} "  # 新增的角度输出
                          f"{pressure_pa:.3e} {pressure_gpa:.3e} "
                          f"{stress_pa[0]:.3e} {stress_pa[1]:.3e} {stress_pa[2]:.3e} "
                          f"{stress_pa[3]:.3e} {stress_pa[4]:.3e} {stress_pa[5]:.3e} "
                          f"{current_pressure_gpa:.3f}")

            # comment_line = f"Lattice parameters: a={a_length:.3f}, b={b_length:.3f}, c={c_length:.3f}, alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}"

            with open(log_filename, "a") as log_file:
                log_file.write(print_stat + "\n")

            # write(xyz_filename, atoms, format='xyz', append=True, comment=comment_line)

            # 更新应力
            current_stress = get_stress_target(current_step)
            current_pressure_gpa = current_stress * eV_A3_to_GPa
            dyn.set_stress(current_stress)


if __name__ == "__main__":
    seed_everything()
    main()

