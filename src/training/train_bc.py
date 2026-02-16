import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from src.models.vlm import MontezumaVLM, Config
from src.data.dataset import AtariDataset, get_transforms
from src.utils.logger import CSVLogger

def train(args):
    # Log metrics to CSV.
    logger = CSVLogger("logs/bc_training.csv")
    global_step = 0

    # Pick the best available device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Model config.
    config = Config(
        img_size=84,
        patch_size=14,
        embed_dim=256,
        n_layers=6,
        n_heads=8,
        seq_len=args.seq_len,
        dropout=0.1,
        frame_stack=4,
        duration_vocab_size=65
    )

    # Dataset.
    transform = get_transforms(img_size=config.img_size)
    full_dataset = AtariDataset(
        data_root='data/atari_v1',
        seq_len=config.seq_len,
        transform=transform,
        limit_trajs=args.limit_trajs,
        subsample_noops=not args.no_noop_subsample,
    )

    # 80/20 train/val split.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Dataset Split: {train_size} Train, {val_size} Val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model + optimizer.
    model = MontezumaVLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1
    )

    criterion_act = nn.CrossEntropyLoss()
    obs_weight = args.obs_weight  # Observation loss weight.

    # Ensure checkpoint directory exists.
    os.makedirs('checkpoints', exist_ok=True)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Observation loss weight (λ): {obs_weight}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_act_loss = 0
        total_obs_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for frames, actions in pbar:
            frames = frames.to(device)
            actions = actions.to(device)

            # Full-sequence forward pass.
            optimizer.zero_grad()
            output = model(frames, actions)

            # Action loss over all timesteps.
            # all_action_logits: (B, T, n_actions), actions: (B, T)
            all_logits = output['all_action_logits']  # (B, T, n_actions)
            B, T, n_actions = all_logits.shape

            # Flatten B and T for CE.
            act_loss = criterion_act(all_logits.reshape(-1, n_actions), actions.reshape(-1))

            # Observation loss over all prediction slots.
            # all_obs_pred: (B, num_images-1, n_patches, embed_dim)
            # Targets are encoded future frame stacks.
            obs_loss = torch.tensor(0.0, device=device)
            all_obs_pred = output['all_obs_pred']
            if all_obs_pred is not None and obs_weight > 0:
                num_obs_preds = all_obs_pred.shape[1]  # Usually 8 (images 1..8).
                # Align targets with prediction slots.
                target_frames = frames[:, 1:num_obs_preds + 1]  # (B, 8, 4, H, W)
                # Encode target frame stacks.
                B_t, N_t, C_t, H_t, W_t = target_frames.shape
                target_frames_flat = target_frames.reshape(B_t * N_t, C_t, H_t, W_t)
                with torch.no_grad():
                    target_embeds = model.patch_embed(target_frames_flat)  # (B*8, n_patches, E)
                target_embeds = target_embeds.view(B_t, N_t, -1, config.embed_dim)  # (B, 8, n_patches, E)

                # Match predicted patch tokens to targets.
                obs_loss = nn.functional.mse_loss(all_obs_pred, target_embeds)

            # Total loss.
            loss = act_loss + obs_weight * obs_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_act_loss += act_loss.item()
            total_obs_loss += obs_loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'act': act_loss.item(),
                'obs': obs_loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })

            global_step += 1
            if global_step % 100 == 0:
                logger.log({
                    'step': global_step,
                    'loss': loss.item(),
                    'act_loss': act_loss.item(),
                    'obs_loss': obs_loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                }, step=global_step)

        avg_train_loss = total_loss / len(train_loader)
        avg_act_loss = total_act_loss / len(train_loader)
        avg_obs_loss = total_obs_loss / len(train_loader)
        
        # Validation.
        model.eval()
        val_loss = 0
        val_act_loss = 0
        val_obs_loss = 0
        val_acc = 0
        total_samples = 0

        with torch.no_grad():
            for frames, actions in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                frames = frames.to(device)
                actions = actions.to(device)

                output = model(frames, actions)

                # Action loss.
                all_logits = output['all_action_logits']
                B, T, n_act = all_logits.shape
                act_loss = criterion_act(all_logits.reshape(-1, n_act), actions.reshape(-1))

                # Observation loss.
                v_obs_loss = torch.tensor(0.0, device=device)
                all_obs_pred = output['all_obs_pred']
                if all_obs_pred is not None and obs_weight > 0:
                    num_obs_preds = all_obs_pred.shape[1]
                    target_frames = frames[:, 1:num_obs_preds + 1]
                    B_t, N_t, C_t, H_t, W_t = target_frames.shape
                    target_frames_flat = target_frames.reshape(B_t * N_t, C_t, H_t, W_t)
                    target_embeds = model.patch_embed(target_frames_flat)
                    target_embeds = target_embeds.view(B_t, N_t, -1, config.embed_dim)
                    v_obs_loss = nn.functional.mse_loss(all_obs_pred, target_embeds)

                loss = act_loss + obs_weight * v_obs_loss

                # Last-step action accuracy.
                pred_act = torch.argmax(all_logits[:, -1, :], dim=-1)
                acc = (pred_act == actions[:, -1]).float().sum().item()

                val_loss += loss.item() * frames.size(0)
                val_act_loss += act_loss.item() * frames.size(0)
                val_obs_loss += v_obs_loss.item() * frames.size(0)
                val_acc += acc
                total_samples += frames.size(0)

        avg_val_loss = val_loss / total_samples
        avg_val_act_loss = val_act_loss / total_samples
        avg_val_obs_loss = val_obs_loss / total_samples
        avg_val_acc = val_acc / total_samples

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} (act={avg_act_loss:.4f}, obs={avg_obs_loss:.4f})")
        print(f"          Val Loss {avg_val_loss:.4f} (act={avg_val_act_loss:.4f}, obs={avg_val_obs_loss:.4f}) | Last-pos Acc {avg_val_acc:.2%}")
        
        logger.log({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'train_act_loss': avg_act_loss,
            'train_obs_loss': avg_obs_loss,
            'val_loss': avg_val_loss,
            'val_act_loss': avg_val_act_loss,
            'val_obs_loss': avg_val_obs_loss,
            'val_acc': avg_val_acc,
        }, step=global_step)

        # Save checkpoint.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'config': config,
        }, f"checkpoints/vlm_epoch_{epoch+1}.pt")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.03)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--limit_trajs', type=float, default=None, help='Percentage of top-scoring trajectories to use (e.g., 20 = top 20%)')
    parser.add_argument('--no_noop_subsample', action='store_true',
                        help='Keep all frames including NOOPs (default: subsample)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--obs_weight', type=float, default=0.1, help='Weight for observation prediction loss (λ in Eq 5)')
    args = parser.parse_args()

    train(args)
