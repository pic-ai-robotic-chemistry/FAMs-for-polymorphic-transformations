import random
from datetime import datetime
from ase import units
from ase.io import write
from ase.md.npt import NPT
from mace.calculators import mace_mp, mace_off
from mace.calculators import MACECalculator
from ase.io.trajectory import Trajectory, TrajectoryReader
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import argparse
import sys

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Run NPT MD simulation with custom pressure ramp and logging intervals. All time arguments are in femtoseconds (fs)."
    )
    parser.add_argument('--file_path', type=str, required=True, help='Path to the CIF file.')
    parser.add_argument('--model', type=str, required=True, help='Path or name of the MACE model.')
    parser.add_argument('--device', type=str, required=True, help='Compute device, e.g. "cpu" or "cuda".')
    parser.add_argument('--xyz_filename', type=str, default='simulation_npt.xyz', help='Output XYZ filename.')
    parser.add_argument('--log_filename', type=str, default='simulation_npt.log', help='Log filename.')
    parser.add_argument('--traj_filename', type=str, default='simulation_npt.traj', help='Trajectory filename.')
    parser.add_argument('--mode', type=str, choices=['increase', 'decrease'], required=True,
                        help='Simulation mode: "increase" or "decrease".')
    parser.add_argument('--total_time_fs', type=float, default=800000.0,
                        help='Total simulation time in femtoseconds (default: 800000 fs = 0.8 ns).')
    parser.add_argument('--ramp_time_fs', type=float, default=100000.0,
                        help='Pressure ramp time in femtoseconds (default: 100000 fs = 0.1 ns).')

    args = parser.parse_args()

    file_path = args.file_path
    model = args.model
    cuda = args.device
    xyz_filename = args.xyz_filename
    log_filename = args.log_filename
    mode = args.mode.lower()
    total_time_fs = float(args.total_time_fs)
    ramp_time_fs = float(args.ramp_time_fs)

    # Read CIF file
    structure = Structure.from_file(file_path)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    print("Original cell vectors:")
    print(atoms.cell)

    # Ensure the cell is upper-triangular
    cell = atoms.cell
    upper_triangular_cell = np.triu(cell)
    atoms.set_cell(upper_triangular_cell, scale_atoms=False)
    print("Modified cell vectors:")
    print(atoms.cell)

    # Create MACE calculator
    calc = mace_mp(model=model, device=cuda, default_dtype="float64")
    atoms.calc = calc

    # Convert user-specified total time (fs) to ASE time units
    timestep = 1.0 * units.fs  # current fixed timestep (1 fs)
    total_time = total_time_fs * units.fs
    # Compute Nsteps robustly
    Nsteps = max(1, int(total_time / timestep))
    print(f"total_time: {total_time} ({total_time_fs} fs)")
    print("timestep:", timestep)
    print("Nsteps:", Nsteps)
    T = 300.0  # Temperature in K

    # Pressure setup: depending on mode, from low to high or high to low
    # Note: original script used these small constants; keep them for backward compatibility.
    if mode == 'increase':
        initial_stress = 0.000101325
        final_stress = 0.01248
    elif mode == 'decrease':
        initial_stress = 0.01248 * 0.5
        final_stress = 0.000101325
    else:
        print("Invalid mode selected.")
        sys.exit(1)

    # Keep ttime and ptime as before (these control barostat/thermostat timing)
    ttime = 100.0 * units.fs
    ptime = 1000.0 * units.fs

    B = 0.6  # eV/Å^3
    pfactor = (ptime / units.fs) ** 2 * B

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

    # Compute ramp_steps from user-specified ramp_time_fs and timestep
    timestep_fs = float(timestep / units.fs)  # typically 1.0
    ramp_steps = max(1, int(ramp_time_fs / timestep_fs))
    # Ensure ramp_steps does not exceed total steps
    if ramp_steps > Nsteps:
        ramp_steps = Nsteps
    print(f"ramp_time: {ramp_time_fs} fs -> ramp_steps: {ramp_steps}")

    def get_stress_target(step):
        if step <= ramp_steps:
            return initial_stress + (final_stress - initial_stress) * (step / float(ramp_steps))
        else:
            return final_stress

    # Prepare log file header
    with open(log_filename, "w") as log_file:
        log_file.write(
            "# step real_time(s) sim_time(fs) epot(J) ekin(J) Temp(K) Etot(J) density(kg/m^3) volume(m^3) a_length b_length c_length alpha beta gamma pressure(Pa) pressure(GPa) pxx(Pa) pyy(Pa) pzz(Pa) pyz(Pa) pxz(Pa) pxy(Pa)\n"
        )

    start_wall_time = datetime.now()

    # For the first 5000 fs (5000 steps) output more frequently
    initial_high_freq_steps = 5000
    high_freq_interval = 1
    low_freq_interval = 1

    eV_to_J = 1.6021765e-19
    A3_to_m3 = 1e-30
    u_to_kg = 1.66053906660e-27
    mass_kg = mass * u_to_kg
    current_step = 0
    eV_A3_to_GPa = 160.21766208
    current_pressure_gpa = initial_stress * eV_A3_to_GPa

    while current_step < Nsteps:
        if current_step < initial_high_freq_steps:
            # High-frequency output for the first 5000 steps: run 1 step per loop
            steps_to_run = 1
            dyn.run(steps_to_run)
            current_step += steps_to_run

            if current_step % high_freq_interval == 0:
                epot = atoms.get_potential_energy()
                ekin = atoms.get_kinetic_energy()
                N = len(atoms)
                temperature = 2.0 * ekin / (3.0 * units.kB * N)
                volume = atoms.get_volume()
                volume_m3 = volume * A3_to_m3
                cell = atoms.get_cell()
                a_length, b_length, c_length = np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
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

                # include angles in the output
                print_stat = (f"{current_step:10d} {real_time.total_seconds():.3f} {sim_time_fs:13.2f} "
                              f"{epot_j:.3e} {ekin_j:.3e} {temperature:.3f} {etot_j:.3e} "
                              f"{density:.3f} {volume_m3:.3e} {a_length:.3f} {b_length:.3f} {c_length:.3f} "
                              f"{alpha:.3f} {beta:.3f} {gamma:.3f} "
                              f"{pressure_pa:.3e} {pressure_gpa:.3e} "
                              f"{stress_pa[0]:.3e} {stress_pa[1]:.3e} {stress_pa[2]:.3e} "
                              f"{stress_pa[3]:.3e} {stress_pa[4]:.3e} {stress_pa[5]:.3e} "
                              f"{current_pressure_gpa:.3f}")
                comment_line = f"Lattice parameters: a={a_length:.3f}, b={b_length:.3f}, c={c_length:.3f}, alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}"

                with open(log_filename, "a") as log_file:
                    log_file.write(print_stat + "\n")

            # Update stress according to ramp
            current_stress = get_stress_target(current_step)
            current_pressure_gpa = current_stress * eV_A3_to_GPa
            dyn.set_stress(current_stress)

        else:
            # Later steps: reduce output frequency (batch runs)
            steps_to_run = low_freq_interval
            if current_step + steps_to_run > Nsteps:
                steps_to_run = Nsteps - current_step

            dyn.run(steps_to_run)
            current_step += steps_to_run

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
                          f"{alpha:.3f} {beta:.3f} {gamma:.3f} "
                          f"{pressure_pa:.3e} {pressure_gpa:.3e} "
                          f"{stress_pa[0]:.3e} {stress_pa[1]:.3e} {stress_pa[2]:.3e} "
                          f"{stress_pa[3]:.3e} {stress_pa[4]:.3e} {stress_pa[5]:.3e} "
                          f"{current_pressure_gpa:.3f}")

            comment_line = f"Lattice parameters: a={a_length:.3f}, b={b_length:.3f}, c={c_length:.3f}, alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}"

            with open(log_filename, "a") as log_file:
                log_file.write(print_stat + "\n")

            # Update stress according to ramp
            current_stress = get_stress_target(current_step)
            current_pressure_gpa = current_stress * eV_A3_to_GPa
            dyn.set_stress(current_stress)


if __name__ == "__main__":
    seed_everything()
    main()
