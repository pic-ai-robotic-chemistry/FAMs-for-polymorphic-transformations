#!/usr/bin/env python3
"""
finetune_sevenn.py
"""

import os
import argparse
from copy import deepcopy

import torch

import sevenn.util as util
from sevenn.nn.scale import SpeciesWiseRescale
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.train.trainer import Trainer
from sevenn.error_recorder import ErrorRecorder
from sevenn.sevenn_logger import Logger

from torch_geometric.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train SevenNet (CLI version of your script).")
    parser.add_argument('--pretrained', type=str, default='7net-0',
                        help="Pretrained name accepted by util.pretrained_name_to_path (default: 7net-0)")
    parser.add_argument('--xyz', type=str, nargs='+', required=True,
                        help="One or more xyz files (space separated) or use shell expansion for a pattern.")
    parser.add_argument('--cutoff', type=float, default=5.0, help="Cutoff radius for graph construction (default: 5.0)")
    parser.add_argument('--processed-name', type=str, default='train.pt', help="Processed dataset filename (default: train.pt)")
    parser.add_argument('--batch-size', type=int, default=20, help="Batch size for training (default: 20)")
    parser.add_argument('--device', type=str, default='cuda:6', help="PyTorch device string (default: cuda:6)")
    parser.add_argument('--epochs', type=int, default=100, help="Total training epochs (default: 100)")
    parser.add_argument('--lr', type=float, default=0.004, help="Learning rate (default: 0.004)")
    parser.add_argument('--scheduler', type=str, default='linearlr', help="Scheduler name (default: linearlr)")
    parser.add_argument('--start-factor', type=float, default=1.0, help="Scheduler start_factor")
    parser.add_argument('--total-iters', type=int, default=10, help="Scheduler total_iters")
    parser.add_argument('--end-factor', type=float, default=0.0001, help="Scheduler end_factor")
    parser.add_argument('--energy-weight', type=float, default=1.0, help="Energy loss weight (default: 1.0)")
    parser.add_argument('--force-weight', type=float, default=25.0, help="Force loss weight (default: 25.0)")
    parser.add_argument('--stress-weight', type=float, default=0.01, help="Stress loss weight (default: 0.01)")
    parser.add_argument('--train-rescale', action='store_true', help="Make rescale (shift/scale) trainable")
    parser.add_argument('--out', type=str, default='linear_lora.pth', help="Output checkpoint base path (default: linear_lora.pth)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument('--split-train', type=float, default=0.90, help="Fraction for training split (default: 0.90)")
    parser.add_argument('--num-workers', type=int, default=0, help="DataLoader num_workers (default: 0)")
    parser.add_argument('--no-screen-log', action='store_true', help="Disable pretty screen logger output")
    return parser.parse_args()


def main():
    args = parse_args()

    # set seed
    torch.manual_seed(args.seed)

    working_dir = os.getcwd()

    # 1) load pretrained model
    print(f"Loading pretrained checkpoint for: {args.pretrained}")
    cp_path = util.pretrained_name_to_path(args.pretrained)
    model, config = util.model_from_checkpoint(cp_path)

    # 2) optionally replace shift-scale module to be trainable
    if args.train_rescale:
        print("Replacing rescale_atomic_energy with trainable SpeciesWiseRescale...")
        shift_scale_module = model._modules.get('rescale_atomic_energy', None)
        if shift_scale_module is None:
            print("Warning: model does not have 'rescale_atomic_energy' module. Skipping replacement.")
        else:
            try:
                shift = shift_scale_module.shift.tolist()
                scale = shift_scale_module.scale.tolist()
            except Exception:
                # fallback if attributes differ
                shift = getattr(shift_scale_module, 'shift').tolist()
                scale = getattr(shift_scale_module, 'scale').tolist()
            model._modules['rescale_atomic_energy'] = SpeciesWiseRescale(
                shift=shift,
                scale=scale,
                train_shift_scale=True,
            )
    else:
        print("Keeping original rescale_atomic_energy module (not trainable).")

    print(model)

    # 3) build dataset (graph preprocessing)
    print("Building/Loading dataset (graph preprocessing)...")
    dataset_files = args.xyz
    dataset = SevenNetGraphDataset(cutoff=args.cutoff, root=working_dir, files=dataset_files, processed_name=args.processed_name)
    print(f"# graphs: {len(dataset)}")
    try:
        print(f"# atoms (nodes): {dataset.natoms}")
    except Exception:
        pass

    if len(dataset) == 0:
        raise RuntimeError("No graphs found in dataset; check xyz files or processed results.")

    # 4) split dataset
    num_dataset = len(dataset)
    num_train = int(num_dataset * args.split_train)
    num_valid = num_dataset - num_train
    dataset = dataset.shuffle()
    train_dataset = dataset[:num_train]
    valid_dataset = dataset[num_train:]
    print(f"# graphs for training: {len(train_dataset)}")
    print(f"# graphs for validation: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 5) Trainer config
    trainer_config = {
        'device': args.device,
        'optimizer': 'adam',
        'optim_param': {'lr': args.lr},
        'scheduler': args.scheduler,
        'scheduler_param': {'start_factor': args.start_factor, 'total_iters': args.total_iters, 'end_factor': args.end_factor},
        'is_ddp': False,
        'force_loss_weight': args.force_loss_weight,
        'stress_loss_weight': args.stress_loss_weight,
        'is_train_stress': False,
    }
    # merge checkpoint config overrides
    config.update(trainer_config)

    trainer = Trainer.from_config(model, config)

    print("Loss functions:", trainer.loss_functions)
    print("Optimizer:", trainer.optimizer)
    print("Scheduler:", trainer.scheduler)

    # 6) Error recorders
    train_recorder = ErrorRecorder.from_config(config)
    valid_recorder = deepcopy(train_recorder)
    for metric in train_recorder.metrics:
        print("Metric:", metric)

    # 7) logger
    logger = Logger()
    logger.screen = not args.no_screen_log

    best_valid = float('inf')

    try:
        with logger:
            logger.greeting()
            for epoch in range(1, args.epochs + 1):
                logger.timer_start('epoch')
                lr_now = trainer.get_lr()
                logger.writeline(f'Epoch {epoch}/{args.epochs}  Learning rate: {lr_now:.6f}')

                # train + validation runs (update recorders)
                trainer.run_one_epoch(train_loader, is_train=True, error_recorder=train_recorder)
                trainer.run_one_epoch(valid_loader, is_train=False, error_recorder=valid_recorder)

                trainer.scheduler_step()

                train_err = train_recorder.epoch_forward()
                valid_err = valid_recorder.epoch_forward()

                logger.bar()
                logger.write_full_table([train_err, valid_err], ['Train', 'Valid'])
                logger.timer_end('epoch', message=f'Epoch {epoch} elapsed')

                # save checkpoint per epoch
                out_base = os.path.abspath(args.out)
                out_dir = os.path.dirname(out_base) if os.path.dirname(out_base) else working_dir
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"{os.path.splitext(os.path.basename(out_base))[0]}_epoch{epoch}.pth"
                out_path = os.path.join(out_dir, out_name)
                trainer.write_checkpoint(out_path, config=config, epoch=epoch)
                print(f"Saved checkpoint: {out_path}")

                # try to compute a numeric validation score to detect best model
                try:
                    valid_score = float(valid_err.get('energy', sum(valid_err.values()) / len(valid_err)))
                except Exception:
                    try:
                        valid_score = float(valid_err)
                    except Exception:
                        valid_score = None

                if valid_score is not None and valid_score < best_valid:
                    best_valid = valid_score
                    best_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(out_base))[0]}_best.pth")
                    trainer.write_checkpoint(best_path, config=config, epoch=epoch)
                    print(f"New best validation ({best_valid}). Saved best checkpoint: {best_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final checkpoint...")
        final_path = os.path.abspath(args.out)
        trainer.write_checkpoint(final_path, config=config, epoch=epoch)
        print(f"Saved checkpoint: {final_path}")
        raise

    # final save
    final_path = os.path.abspath(args.out)
    trainer.write_checkpoint(final_path, config=config, epoch=args.epochs)
    print(f"Training completed. Final checkpoint saved: {final_path}")


if __name__ == '__main__':
    main()
