
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
from tqdm import tqdm
import numpy as np
from collections import deque
from time import perf_counter
from PIL import Image
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from src.models.vlm import MontezumaVLM, Config
from src.models.metacontroller import Metacontroller
from src.data.dataset import AtariDataset, get_transforms
from src.utils.logger import CSVLogger


def evaluate(base_model, metacontroller, val_loader, device, control_layer, fixed_switch_rate=None):
    """Run validation for the metacontroller."""
    metacontroller.eval()
    ignore_action_id = base_model.start_token_id

    total_nll, total_kl, total_samples = 0, 0, 0
    all_betas = []
    all_z_stds = []
    all_kl = []
    all_mu_norms = []
    all_logvars = []
    all_beta_frac = []
    all_beta_switches = []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                frames, actions = batch
            else:
                frames = batch['frames']
                actions = batch['actions']
            frames = frames.to(device)
            actions = actions.to(device)

            # Extract residuals at the action-token positions.
            output_1 = base_model(frames, actions, return_residuals=True)
            action_positions_t = torch.as_tensor(
                output_1['action_positions'], device=device, dtype=torch.long
            )
            assert action_positions_t.numel() == actions.shape[1]
            residuals_full = output_1['residuals'][control_layer]
            residuals = residuals_full[:, action_positions_t, :]

            # Metacontroller forward.
            meta_out = metacontroller(residuals, training=True, fixed_switch_rate=fixed_switch_rate)

            # Build adapter tensors.
            if meta_out.get('us_stacked') is not None:
                A_tensor, B_tensor = meta_out['us_stacked']
            else:
                A_list = [u[0] for u in meta_out['us']]
                B_list = [u[1] for u in meta_out['us']]
                A_tensor = torch.stack(A_list, dim=1)
                B_tensor = torch.stack(B_list, dim=1)
            if A_tensor.shape[1] != residuals_full.shape[1]:
                A_full = torch.zeros(
                    A_tensor.shape[0], residuals_full.shape[1], A_tensor.shape[2], A_tensor.shape[3],
                    device=A_tensor.device, dtype=A_tensor.dtype
                )
                B_full = torch.zeros_like(A_full)
                A_full[:, action_positions_t] = A_tensor
                B_full[:, action_positions_t] = B_tensor
                A_tensor, B_tensor = A_full, B_full

            # KL term.
            mus, logvars = meta_out['mus'], meta_out['logvars']
            kl = meta_out['kl'].mean()

            # Base model forward with adapters.
            output_2 = base_model(frames, actions,
                                  adapter_params=(A_tensor, B_tensor, control_layer))

            # Action NLL.
            all_logits = output_2['all_action_logits']
            B_sz, T_sz, n_act = all_logits.shape
            nll = F.cross_entropy(
                all_logits.reshape(-1, n_act),
                actions.reshape(-1),
                ignore_index=ignore_action_id,
            )

            batch_size = frames.size(0)
            total_nll += nll.item() * batch_size
            total_kl += kl.item() * batch_size
            total_samples += batch_size

            # Beta statistics.
            betas = meta_out['betas']
            all_betas.append(betas.mean().item())
            all_beta_frac.append((betas > 0.5).float().mean().item())
            all_beta_switches.append((betas > 0.5).float().sum(dim=1).mean().item())
            all_z_stds.append(meta_out['zs'].std(dim=0).mean().item())
            all_kl.append(meta_out['kl'])
            all_mu_norms.append(mus.norm(dim=-1).mean().item())
            all_logvars.append(logvars.detach())

    metacontroller.train()

    kl_cat = torch.cat([k.reshape(-1) for k in all_kl]).detach().cpu() if all_kl else torch.tensor([0.0])
    logvars_cat = torch.cat([lv.reshape(-1) for lv in all_logvars]).detach().cpu() if all_logvars else torch.tensor([0.0])

    return {
        'val_nll': total_nll / total_samples,
        'val_kl': total_kl / total_samples,
        'avg_beta': np.mean(all_betas),
        'beta_switch_rate': np.mean(all_beta_frac) if all_beta_frac else 0.0,
        'beta_switches_per_traj': np.mean(all_beta_switches) if all_beta_switches else 0.0,
        'z_diversity': np.mean(all_z_stds),
        'kl_mean': kl_cat.mean().item(),
        'kl_p95': kl_cat.quantile(0.95).item(),
        'kl_max': kl_cat.max().item(),
        'mu_norm_mean': np.mean(all_mu_norms) if all_mu_norms else 0.0,
        'logvar_mean': logvars_cat.mean().item(),
        'logvar_min': logvars_cat.min().item(),
        'logvar_max': logvars_cat.max().item(),
    }




def train_meta(args):
    # Create a timestamped log file.
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(f"logs/meta_{timestamp}.csv")
    global_step = 0

    # Device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    def _sync_if_needed():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # Model config.
    config = Config(
        img_size=84,
        patch_size=14,
        embed_dim=256,
        n_layers=6,
        n_heads=8,
        seq_len=args.seq_len,
        dropout=0.1,
        frame_stack=4
    )

    # Load base model.
    base_model = MontezumaVLM(config).to(device)
    if args.checkpoint:
        print(f"Loading base model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: Training on random base model")

    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    ignore_action_id = base_model.start_token_id

    # Initialize metacontroller.
    control_layer = config.n_layers // 2
    metacontroller = Metacontroller(
        config, config.embed_dim,
        aux_position_predictor=(args.position_weight > 0)
    ).to(device)

    optimizer = optim.AdamW(metacontroller.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume if requested.
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # Support both old and new checkpoint formats.
        if 'metacontroller_state_dict' in resume_ckpt:
            metacontroller.load_state_dict(resume_ckpt['metacontroller_state_dict'])
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            start_epoch = resume_ckpt['epoch']
            global_step = resume_ckpt['global_step']
            print(f"Resumed from epoch {start_epoch}, global_step {global_step}")
        else:
            # Old format: raw state_dict only.
            metacontroller.load_state_dict(resume_ckpt)
            print("Resumed from old-format checkpoint (epoch/step unknown, starting from 0)")

    # Dataset.
    transform = get_transforms(img_size=config.img_size)
    if args.position_weight > 0:
        print("Warning: AtariDataset does not include position labels. Disabling position loss.")
        args.position_weight = 0.0
    dataset = AtariDataset(
        data_root='data/atari_v1',
        seq_len=config.seq_len,
        transform=transform,
        limit_trajs=args.limit_trajs,
        subsample_noops=not args.no_noop_subsample,
    )
    noop_status = "disabled" if args.no_noop_subsample else "enabled (10% keep)"
    print(f"\nUsing AtariDataset (top {args.limit_trajs}% by score): {len(dataset)} samples")
    print(f"  NOOP subsampling: {noop_status}")

    # 80/20 train/val split.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Split: {train_size} train, {val_size} val")

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  KL weight (α): {args.kl_weight}")
    print(f"  Position weight: {args.position_weight}" + (" [hinged]" if args.hinged_position_loss else ""))
    if args.switch_rate is not None:
        print(f"  Fixed switch rate: {args.switch_rate} frames (bypasses learned beta)")

    for epoch in range(start_epoch, args.epochs):
        metacontroller.train()
        total_loss, total_nll, total_kl, total_pos_loss = 0, 0, 0, 0
        total_beta, total_switches = 0, 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        prev_iter_end = perf_counter()
        for batch in pbar:
            now = perf_counter()
            data_time = now - prev_iter_end
            if isinstance(batch, (tuple, list)):
                frames, actions = batch
                positions = None
            else:
                frames = batch['frames']
                actions = batch['actions']
                positions = batch.get('positions')
            frames = frames.to(device)
            actions = actions.to(device)
            positions = positions.to(device) if positions is not None else None


            with torch.no_grad():
                output_1 = base_model(frames, actions, return_residuals=True)
                action_positions_t = torch.as_tensor(
                    output_1['action_positions'], device=device, dtype=torch.long
                )
                assert action_positions_t.numel() == actions.shape[1]
                residuals_full = output_1['residuals'][control_layer]
                residuals = residuals_full[:, action_positions_t, :]
            seq_context = None

            

            # Metacontroller forward.
            meta_out = metacontroller(residuals, training=True, seq_context=seq_context,
                                      fixed_switch_rate=args.switch_rate)
            

            # Build adapter tensors.
            if meta_out.get('us_stacked') is not None:
                A_tensor, B_tensor = meta_out['us_stacked']
            else:
                A_tensor, B_tensor = map(lambda xs: torch.stack(xs, 1), zip(*meta_out["us"]))
            if A_tensor.shape[1] != residuals_full.shape[1]:
                A_full = torch.zeros(
                    A_tensor.shape[0], residuals_full.shape[1], A_tensor.shape[2], A_tensor.shape[3],
                    device=A_tensor.device, dtype=A_tensor.dtype
                )
                B_full = torch.zeros_like(A_full)
                A_full[:, action_positions_t] = A_tensor
                B_full[:, action_positions_t] = B_tensor
                A_tensor, B_tensor = A_full, B_full
            


            # KL loss.
            B_batch, T_seq = residuals.shape[:2]
            mus = meta_out['mus']
            logvars = meta_out['logvars']
            kl_loss = meta_out['kl'].mean()

            # Base model forward with adapters.
            output_2 = base_model(frames, actions,
                                  adapter_params=(A_tensor, B_tensor, control_layer))
            

            # Action NLL.
            all_logits = output_2['all_action_logits']
            B_sz, T_sz, n_act = all_logits.shape
            nll_loss = F.cross_entropy(
                all_logits.reshape(-1, n_act),
                actions.reshape(-1),
                ignore_index=ignore_action_id,
            )

            # Beta stats.
            betas = meta_out['betas']
            avg_beta = betas.mean().item()
            switch_rate = (betas > 0.5).float().mean().item()

            # KL annealing.
            if args.kl_anneal_steps > 0:
                alpha = args.kl_weight * min(1.0, global_step / args.kl_anneal_steps)
            else:
                alpha = args.kl_weight

            # Optional position loss on z.
            position_loss = torch.tensor(0.0, device=device)
            if args.position_weight > 0 and positions is not None and 'position_preds' in meta_out:
                position_preds = meta_out['position_preds']  # (B, T_meta, 2)
                T_pos = positions.shape[1]  # 64 action timesteps
                T_meta = position_preds.shape[1]  # 388 VLM sequence tokens

                # Match prediction length to action length.
                if T_meta != T_pos:
                    indices = torch.linspace(0, T_meta - 1, T_pos).long().to(device)
                    position_preds = position_preds[:, indices, :]  # Now (B, 64, 2)

                # Ignore timesteps with missing position labels.
                valid_mask = (positions >= 0).all(dim=-1)  # (B, T)

                if args.hinged_position_loss:
                    # Hinged position loss: per-timestep target.
                    if valid_mask.any():
                        # Per-timestep squared error.
                        per_t_loss = (position_preds - positions).pow(2).sum(dim=-1)  # (B, T)
                        position_loss = per_t_loss[valid_mask].mean()
                else:
                    # Standard position MSE.
                    if valid_mask.any():
                        position_loss = F.mse_loss(
                            position_preds[valid_mask],
                            positions[valid_mask]
                        )

            # Total loss.
            loss = (nll_loss + alpha * kl_loss +
                    args.position_weight * position_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nll += nll_loss.item()
            total_kl += kl_loss.item()
            total_pos_loss += position_loss.item()
            total_beta += avg_beta
            total_switches += switch_rate
            n_batches += 1

            postfix = {
                'loss': f'{loss.item():.3f}',
                'nll': f'{nll_loss.item():.3f}',
                'kl': f'{kl_loss.item():.3f}',
                'β': f'{avg_beta:.3f}',
                'sw': f'{switch_rate:.1%}'
            }
            if args.position_weight > 0:
                postfix['pos'] = f'{position_loss.item():.4f}'
            pbar.set_postfix(postfix)

            # Periodic metrics logging.
            if global_step % 100 == 0:
                beta_switches_per_traj = (betas > 0.5).float().sum(dim=1).mean().item()
                mu_norm_mean = mus.norm(dim=-1).mean().item()
                logvar_mean = logvars.mean().item()
                logvar_min = logvars.min().item()
                logvar_max = logvars.max().item()
                kl_flat = meta_out['kl'].detach().cpu().reshape(-1)
                kl_mean = kl_flat.mean().item()
                kl_p95 = kl_flat.quantile(0.95).item()
                kl_max = kl_flat.max().item()
                with torch.no_grad():
                    # Baseline model output without adapters.
                    output_uncontrolled = base_model(frames, actions)
                    adapter_effect = (output_2['logits'] - output_uncontrolled['logits']).abs().mean().item()

                    # Baseline NLL on clean frames.
                    base_logits_clean = output_uncontrolled['all_action_logits']
                    base_nll_clean = F.cross_entropy(
                        base_logits_clean.reshape(-1, n_act),
                        actions.reshape(-1),
                        ignore_index=ignore_action_id,
                    ).item()

                    # Placeholder: degraded-frame baseline not computed separately.
                    base_nll_degraded = base_nll_clean

                print(f"\n[Step {global_step}] Adapter: {adapter_effect:.4f} | β: {avg_beta:.3f} | KL_α: {alpha:.4f}")
                
                logger.log({
                    'step': global_step,
                    'loss': loss.item(),
                    'action_nll': nll_loss.item(),
                    'kl': kl_loss.item(),
                    'kl_mean': kl_mean,
                    'kl_p95': kl_p95,
                    'kl_max': kl_max,
                    'position_loss': position_loss.item(),
                    'alpha': alpha,
                    'beta_mean': avg_beta,
                    'beta_frac_gt_0.5': switch_rate,
                    'beta_switches_per_traj': beta_switches_per_traj,
                    'mu_norm_mean': mu_norm_mean,
                    'logvar_mean': logvar_mean,
                    'logvar_min': logvar_min,
                    'logvar_max': logvar_max,
                    'adapter_effect': adapter_effect,
                    'base_nll_clean': base_nll_clean,
                    'base_nll_degraded': base_nll_degraded,
                    }, step=global_step)

            global_step += 1
            prev_iter_end = perf_counter()

        # Epoch summary.
        avg_loss = total_loss / n_batches
        avg_beta = total_beta / n_batches

        # Validation.
        val_metrics = evaluate(base_model, metacontroller, val_loader, device, control_layer, args.switch_rate)
        print(f"\nEpoch {epoch+1}: Loss {avg_loss:.4f} | Val NLL: {val_metrics['val_nll']:.4f} | Val KL: {val_metrics['val_kl']:.4f}")
        print(f"  β: {val_metrics['avg_beta']:.4f} | Z diversity: {val_metrics['z_diversity']:.4f}")

        # Save checkpoint.
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'metacontroller_state_dict': metacontroller.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"checkpoints/meta_simple_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.03)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--kl_weight', type=float, default=0.1, help="KL weight (α)")
    parser.add_argument('--kl_anneal_steps', type=int, default=5000, help="Steps to anneal KL from 0 to kl_weight")
    parser.add_argument('--max_frames', type=int, default=500, help="Max frames for trajectory (filters dataset)")
    parser.add_argument('--dataset_type', type=str, default='atari', choices=['atari', 'enhanced'],
                        help="Deprecated. Ignored; AtariDataset is always used.")
    parser.add_argument('--limit_trajs', type=float, default=5,
                        help="Percentage of top-scoring trajectories to use")
    parser.add_argument('--checkpoint', type=str, default=None, help="Base model checkpoint")

    parser.add_argument('--num_workers', type=int, default=0)

    # Position-loss options.
    parser.add_argument('--position_weight', type=float, default=0.0,
                        help="Position prediction loss weight (forces z to carry spatial info)")
    parser.add_argument('--hinged_position_loss', action='store_true',
                        help="Only compute position loss at current timestep (forces z refresh on movement)")

    # Fixed-switching option.
    parser.add_argument('--switch_rate', type=int, default=None,
                        help="Force z switch every N frames (bypasses learned beta)")
    
    # Keep all NOOPs instead of subsampling.
    parser.add_argument('--no_noop_subsample', action='store_true',
                        help="Keep all frames including NOOPs (default: subsample to 10%)")

    # Resume checkpoint path.
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    train_meta(args)
