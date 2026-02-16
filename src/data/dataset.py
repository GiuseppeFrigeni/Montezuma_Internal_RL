import glob
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def _ensure_tensor_frame(img):
    """Ensure a frame is a CHW float tensor."""
    if torch.is_tensor(img):
        return img
    return T.ToTensor()(img)


def get_transforms(img_size=84):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])


class AtariDataset(Dataset):
    def __init__(
        self,
        data_root,
        seq_len=64,
        transform=None,
        limit_trajs=None,
        subsample_noops=True,
    ):
        """
        Unified Atari dataset used by both BC and metacontroller training.

        Args:
            data_root: Path to data directory.
            seq_len: Sequence length for samples.
            transform: Image transforms.
            limit_trajs: Percentage of top-scoring trajectories to use.
            subsample_noops: If True, subsample NOOP actions. If False, keep all.
        """
        self.data_root = data_root
        self.seq_len = seq_len
        self.transform = transform
        self.subsample_noops = subsample_noops

        self.traj_dir = os.path.join(data_root, "trajectories", "revenge")
        self.screen_dir = os.path.join(data_root, "screens", "revenge")
        self.samples = []

        traj_files = sorted(glob.glob(os.path.join(self.traj_dir, "*.txt")))
        print(f"Scanning {len(traj_files)} trajectories for scores...")

        scored_files = []
        for tf in traj_files:
            try:
                _, _, _, _, final_score = self._load_trajectory(tf)
                scored_files.append((tf, final_score))
            except Exception:
                continue

        scored_files.sort(key=lambda x: x[1], reverse=True)

        if limit_trajs is not None and len(scored_files) > 0:
            n_keep = max(1, int(len(scored_files) * limit_trajs / 100))
            print(
                f"Selecting top {limit_trajs}% trajectories by score "
                f"({n_keep}/{len(scored_files)})"
            )
            scored_files = scored_files[:n_keep]

        print(f"Indexing {len(scored_files)} trajectories with death-free segments...")

        for tf, _score in scored_files:
            traj_id = os.path.basename(tf).replace(".txt", "")
            try:
                raw_frames, raw_actions, rewards, terminals, _ = self._load_trajectory(tf)
                self._process_death_free_segments(
                    traj_id, raw_frames, raw_actions, rewards, terminals
                )
            except Exception:
                continue

        print(f"Found {len(self.samples)} samples.")

    def _load_trajectory(self, traj_file):
        df = pd.read_csv(traj_file, skiprows=1)
        if "frame" not in df.columns:
            df = pd.read_csv(
                traj_file,
                skiprows=2,
                names=["frame", "reward", "score", "terminal", "action"],
            )

        df.columns = [c.strip() for c in df.columns]

        raw_frames = df["frame"].to_numpy()
        raw_actions = df["action"].to_numpy()

        if "reward" in df.columns:
            rewards = df["reward"].to_numpy()
        else:
            rewards = np.zeros(len(raw_frames), dtype=np.float32)

        if "terminal" in df.columns:
            term_col = df["terminal"]
            if term_col.dtype == object:
                terminals = (term_col.str.strip().str.lower() == "true").to_numpy()
            else:
                terminals = term_col.to_numpy().astype(bool)
        else:
            terminals = np.zeros(len(raw_frames), dtype=bool)

        if "score" in df.columns:
            final_score = float(df["score"].max())
        else:
            final_score = float(np.sum(rewards))

        return raw_frames, raw_actions, rewards, terminals, final_score

    def _subsample_sequence(self, frames, actions, noop_keep_prob):
        keep_mask = []
        for act in actions:
            if act != 0:
                keep_mask.append(True)
            else:
                keep_mask.append(np.random.random() < noop_keep_prob)
        return frames[keep_mask], actions[keep_mask]

    def _append_fixed_windows(self, traj_id, frames, actions):
        if len(actions) == 0:
            return

        start_token = 18
        pad_len = self.seq_len
        pad_actions = np.full(pad_len, start_token, dtype=actions.dtype)
        pad_frames = np.repeat(frames[:1], pad_len, axis=0)

        actions = np.concatenate([pad_actions, actions])
        frames = np.concatenate([pad_frames, frames])

        num_frames = len(actions)
        if num_frames <= self.seq_len:
            return

        for start_idx in range(0, num_frames - self.seq_len):
            target_action = actions[start_idx + self.seq_len - 1]

            keep_sample = False
            if start_idx < pad_len:
                keep_sample = True
            elif target_action != 0:
                keep_sample = True
            elif not self.subsample_noops:
                keep_sample = True
            elif np.random.random() < 0.05:
                keep_sample = True

            if keep_sample:
                self.samples.append(
                    {
                        "traj_id": traj_id,
                        "frames": frames[start_idx : start_idx + self.seq_len],
                        "actions": actions[start_idx : start_idx + self.seq_len],
                    }
                )

    def _process_death_free_segments(self, traj_id, raw_frames, raw_actions, rewards, terminals):
        reward_indices = np.where(rewards > 0)[0]
        if len(reward_indices) == 0:
            return

        death_indices = set(np.where(terminals)[0])
        segment_starts = [0] + list(reward_indices[:-1] + 1)
        segment_ends = list(reward_indices)

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            if seg_end <= seg_start:
                continue

            segment_deaths = death_indices & set(range(seg_start, seg_end))
            if len(segment_deaths) > 0:
                continue

            seg_frames = raw_frames[seg_start : seg_end + 1]
            seg_actions = raw_actions[seg_start : seg_end + 1]

            if len(seg_actions) < 10:
                continue

            if self.subsample_noops:
                # Meta dataset historically used 10% keep for NOOPs.
                seg_frames, seg_actions = self._subsample_sequence(
                    seg_frames, seg_actions, noop_keep_prob=0.10
                )

            if len(seg_actions) < 5:
                continue

            self._append_fixed_windows(traj_id, seg_frames, seg_actions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        traj_id = sample["traj_id"]
        frame_indices = sample["frames"]
        actions = sample["actions"]

        snapshot_steps = [i * 8 for i in range(8)] + [self.seq_len - 1]
        max_step = self.seq_len - 1

        multi_stacks = []
        for step_idx in snapshot_steps:
            step_idx = max(0, min(max_step, step_idx))
            game_frame_idx = frame_indices[step_idx]

            stack_indices = [game_frame_idx - i for i in range(3, -1, -1)]
            frames = []
            for f_idx in stack_indices:
                if f_idx < 1:
                    f_idx = 1
                img_path = os.path.join(self.screen_dir, traj_id, f"{f_idx}.png")
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("L")
                        if self.transform:
                            img = self.transform(img)
                        frames.append(_ensure_tensor_frame(img))
                except FileNotFoundError:
                    if frames:
                        frames.append(frames[-1].clone())
                    else:
                        frames.append(torch.zeros((1, 84, 84), dtype=torch.float32))

            multi_stacks.append(torch.cat(frames, dim=0))

        frames_tensor = torch.stack(multi_stacks)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        return frames_tensor, actions_tensor
