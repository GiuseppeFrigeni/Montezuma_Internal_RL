#!/usr/bin/env python3
"""Evaluate VLM/metacontroller checkpoints on ALE/MontezumaRevenge-v5.

Modes:
- No agent checkpoint: train_meta-style latent generation via Metacontroller.forward.
- With agent checkpoint: internal-rl notebook-style latent generation via policy z.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import ale_py


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.vlm import Config, MontezumaVLM
from src.models.metacontroller import HawkBlock, Metacontroller


gym.register_envs(ale_py)


class InternalRLWrapper(gym.Env):
    """Environment preprocessing and history buffers as in internal-rl notebook."""

    def __init__(self, device: torch.device, seq_len: int, render_mode: Optional[str] = None):
        super().__init__()
        self.device = device
        self.env = gym.make("ALE/MontezumaRevenge-v5", render_mode=render_mode, frameskip=1)

        self.seq_len = seq_len
        self.num_images = 9
        self.frame_stack = 4

        self.frame_buffer = deque(maxlen=self.num_images * self.frame_stack)
        self.action_history = deque(maxlen=self.seq_len - 1)

    def close(self):
        self.env.close()

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        self._reset_buffers(obs)
        return obs, info

    def primitive_step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self._update_frame_buffer(obs)
        self.action_history.append(int(action))
        return reward, terminated, truncated, info

    def _reset_buffers(self, obs: np.ndarray):
        self.frame_buffer.clear()
        self.action_history.clear()

        frame = self._process_frame(obs)
        for _ in range(self.num_images * self.frame_stack):
            self.frame_buffer.append(frame)

        for _ in range(self.seq_len - 1):
            self.action_history.append(0)

    def _update_frame_buffer(self, obs: np.ndarray):
        self.frame_buffer.append(self._process_frame(obs))

    def _process_frame(self, obs: np.ndarray) -> torch.Tensor:
        img = cv2.resize(obs, (84, 84))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return torch.tensor(img, dtype=torch.float32, device=self.device) / 255.0

    def get_model_inputs(self):
        frames_list = list(self.frame_buffer)
        actions_list = list(self.action_history) + [0]

        images = []
        for i in range(self.num_images):
            start = i * self.frame_stack
            images.append(torch.stack(frames_list[start:start + self.frame_stack], dim=0))

        frames = torch.stack(images, dim=0).unsqueeze(0)  # (1, 9, 4, 84, 84)
        actions = torch.tensor(actions_list, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)
        return frames, actions

    def get_current_embedding(self, base_model: MontezumaVLM, control_layer: int) -> np.ndarray:
        frames, actions = self.get_model_inputs()
        with torch.no_grad():
            out = base_model(frames, actions, return_residuals=True)
            lap = int(out["action_positions"][-1])
            e = out["residuals"][control_layer][:, lap, :]
        return e.detach().cpu().numpy().flatten()


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Same policy architecture as internal-rl notebook."""

    def __init__(self, input_dim: int = 256, latent_dim: int = 8, hidden_dim: int = 256, initial_std: float = 0.7):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.ssm = HawkBlock(hidden_dim, d_state=256, n_heads=8, mlp_ratio=2)

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, latent_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.ones(latent_dim) * np.log(initial_std))

    def step(self, x_t: torch.Tensor, state=None):
        x_t = self.input_proj(x_t)
        x_t, new_state = self.ssm.step(x_t, state)
        mu = self.actor_mean(x_t)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return Normal(mu, std), new_state


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_state_dict(ckpt_obj, preferred_key: str):
    if isinstance(ckpt_obj, dict) and preferred_key in ckpt_obj:
        return ckpt_obj[preferred_key]
    return ckpt_obj


def _config_from_checkpoint(ckpt: dict) -> Config:
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(cfg, Config):
        return cfg
    if is_dataclass(cfg):
        return Config(**asdict(cfg))
    if isinstance(cfg, dict):
        return Config(**cfg)
    return Config()


def load_vlm(vlm_checkpoint: str, device: torch.device):
    ckpt = torch.load(vlm_checkpoint, map_location=device, weights_only=False)
    config = _config_from_checkpoint(ckpt)

    model = MontezumaVLM(config).to(device).eval()
    model_sd = _extract_state_dict(ckpt, "model_state_dict")
    model.load_state_dict(model_sd)
    return model, config


def _infer_meta_init_kwargs(meta_state_dict: dict):
    aux_position_predictor = any(k.startswith("position_predictor.") for k in meta_state_dict.keys())
    waypoint_conditioned = any(k.startswith("waypoint_embedding.") for k in meta_state_dict.keys())
    num_waypoints = 6
    if waypoint_conditioned and "waypoint_embedding.weight" in meta_state_dict:
        num_waypoints = int(meta_state_dict["waypoint_embedding.weight"].shape[0])
    return aux_position_predictor, waypoint_conditioned, num_waypoints


def load_metacontroller(meta_checkpoint: str, config: Config, device: torch.device):
    ckpt = torch.load(meta_checkpoint, map_location=device, weights_only=False)
    meta_sd = _extract_state_dict(ckpt, "metacontroller_state_dict")

    aux_position_predictor, waypoint_conditioned, num_waypoints = _infer_meta_init_kwargs(meta_sd)
    metacontroller = Metacontroller(
        config,
        config.embed_dim,
        aux_position_predictor=aux_position_predictor,
        waypoint_conditioned=waypoint_conditioned,
        num_waypoints=num_waypoints,
    ).to(device).eval()

    missing, unexpected = metacontroller.load_state_dict(meta_sd, strict=False)
    if missing:
        print(f"[warn] Missing metacontroller keys: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected metacontroller keys: {len(unexpected)}")
    return metacontroller


def load_agent(agent_checkpoint: str, device: torch.device):
    ckpt = torch.load(agent_checkpoint, map_location=device, weights_only=False)
    agent_sd = _extract_state_dict(ckpt, "policy_state_dict")

    agent = Actor().to(device).eval()
    agent.load_state_dict(agent_sd)
    return agent


def compute_action_positions(num_images: int, n_patches: int, seq_len: int, chunk: int = 8):
    action_positions = []
    cur = 0
    for i in range(num_images):
        cur += n_patches
        start = i * chunk
        end = min((i + 1) * chunk, seq_len)
        if start < seq_len:
            k = end - start
            action_positions.extend([cur + j for j in range(k)])
            cur += k
    total_tokens = num_images * n_patches + seq_len
    return action_positions, total_tokens


def build_tokenwise_adapters_from_z(
    curr_z: torch.Tensor,
    metacontroller: Metacontroller,
    config: Config,
    num_images: int,
    seq_len: int,
    device: torch.device,
):
    n_envs = curr_z.shape[0]
    d_model = config.embed_dim
    rank = metacontroller.rank
    n_patches = (config.img_size // config.patch_size) ** 2

    action_positions, total_tokens = compute_action_positions(num_images, n_patches, seq_len)
    idx = torch.tensor(action_positions, device=device, dtype=torch.long)

    params = metacontroller.hypernet(curr_z).view(n_envs, 2, d_model, rank)
    a0 = params[:, 0]
    b0 = params[:, 1]

    a_seq = torch.zeros(n_envs, total_tokens, d_model, rank, device=device)
    b_seq = torch.zeros_like(a_seq)
    a_seq[:, idx] = a0.unsqueeze(1).expand(n_envs, idx.numel(), d_model, rank)
    b_seq[:, idx] = b0.unsqueeze(1).expand(n_envs, idx.numel(), d_model, rank)
    return a_seq, b_seq


def sample_action(logits: torch.Tensor, deterministic: bool = False) -> int:
    if deterministic:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def run_episode_train_meta_mode(
    env: InternalRLWrapper,
    base_model: MontezumaVLM,
    metacontroller: Metacontroller,
    config: Config,
    control_layer: int,
    max_steps: Optional[int],
    deterministic: bool,
    switch_rate: Optional[int],
    seed: Optional[int] = None,
):
    env.reset(seed=seed)
    episode_return = 0.0
    steps = 0
    beta_values = []

    done = False
    while not done:
        if max_steps is not None and steps >= max_steps:
            break

        frames, actions = env.get_model_inputs()

        with torch.no_grad():
            out_clean = base_model(frames, actions, return_residuals=True)
            action_positions = torch.as_tensor(out_clean["action_positions"], device=frames.device, dtype=torch.long)
            residuals_full = out_clean["residuals"][control_layer]
            residuals = residuals_full[:, action_positions, :]

            meta_out = metacontroller(
                residuals,
                training=True,
                fixed_switch_rate=switch_rate,
            )
            a_actions, b_actions = meta_out["us_stacked"]

            total_tokens = residuals_full.shape[1]
            d_model = config.embed_dim
            rank = metacontroller.rank

            a_full = torch.zeros(1, total_tokens, d_model, rank, device=frames.device)
            b_full = torch.zeros_like(a_full)
            a_full[:, action_positions] = a_actions
            b_full[:, action_positions] = b_actions

            out_ctrl = base_model(frames, actions, adapter_params=(a_full, b_full, control_layer))
            action = sample_action(out_ctrl["logits"].squeeze(0), deterministic=deterministic)

            beta_last = float(meta_out["betas"][:, -1, :].mean().item())
            beta_values.append(beta_last)

        reward, terminated, truncated, _ = env.primitive_step(action)
        done = terminated or truncated
        episode_return += float(reward)
        steps += 1

    return {
        "return": episode_return,
        "steps": steps,
        "done": done,
        "beta_mean": float(np.mean(beta_values)) if beta_values else float("nan"),
        "switches": None,
    }


def run_episode_agent_mode(
    env: InternalRLWrapper,
    base_model: MontezumaVLM,
    metacontroller: Metacontroller,
    agent: Actor,
    config: Config,
    control_layer: int,
    beta_threshold: float,
    max_steps: Optional[int],
    deterministic: bool,
    switch_rate: Optional[int],
    seed: Optional[int] = None,
):
    env.reset(seed=seed)

    h_switch = torch.zeros(1, metacontroller.n_h, device=env.device)

    obs = env.get_current_embedding(base_model, control_layer)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=env.device).unsqueeze(0)
    with torch.no_grad():
        dist_z, h_policy = agent.step(obs_t, None)
        curr_z = dist_z.mean if deterministic else dist_z.sample()

    episode_return = 0.0
    steps = 0
    switches = 0
    switch_events = []
    beta_values = []
    option_prim_steps = 0
    done = False

    while not done:
        if max_steps is not None and steps >= max_steps:
            break

        frames, actions = env.get_model_inputs()

        with torch.no_grad():
            out_clean = base_model(frames, actions, return_residuals=True)
            lap = int(out_clean["action_positions"][-1])
            e_t_clean = out_clean["residuals"][control_layer][:, lap, :]

            switch_in = torch.cat([e_t_clean, h_switch, curr_z], dim=1)
            beta_logit = metacontroller.switch_net(switch_in)
            beta = torch.sigmoid(beta_logit)
            h_switch = metacontroller.history_gru(e_t_clean, h_switch)

            a_seq, b_seq = build_tokenwise_adapters_from_z(
                curr_z=curr_z,
                metacontroller=metacontroller,
                config=config,
                num_images=env.num_images,
                seq_len=env.seq_len,
                device=env.device,
            )

            out_ctrl = base_model(
                frames,
                actions,
                return_residuals=False,
                adapter_params=(a_seq, b_seq, control_layer),
            )
            action = sample_action(out_ctrl["logits"].squeeze(0), deterministic=deterministic)

        reward, terminated, truncated, _ = env.primitive_step(action)
        done = terminated or truncated
        episode_return += float(reward)
        steps += 1
        option_prim_steps += 1

        beta_val = float(beta.item())
        beta_values.append(beta_val)
        if switch_rate is not None:
            ended_by_switch = (option_prim_steps >= switch_rate) and (not done)
        else:
            ended_by_switch = (beta_val > beta_threshold) and (not done)

        if ended_by_switch:
            switches += 1
            switch_events.append(
                {
                    "switch_idx": switches,
                    "frame": steps,
                    "beta": beta_val,
                }
            )
            option_prim_steps = 0
            obs = env.get_current_embedding(base_model, control_layer)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=env.device).unsqueeze(0)
            with torch.no_grad():
                dist_z, h_policy = agent.step(obs_t, h_policy)
                curr_z = dist_z.mean if deterministic else dist_z.sample()

    return {
        "return": episode_return,
        "steps": steps,
        "done": done,
        "beta_mean": float(np.mean(beta_values)) if beta_values else float("nan"),
        "switches": switches,
        "switch_events": switch_events,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Play MontezumaRevenge with VLM + metacontroller (+ optional agent).")
    parser.add_argument("--vlm-checkpoint", type=str, required=True, help="Path to VLM checkpoint.")
    parser.add_argument("--metacontroller-checkpoint", type=str, required=True, help="Path to metacontroller checkpoint.")
    parser.add_argument("--agent-checkpoint", type=str, default=None, help="Optional agent checkpoint (internal-rl mode).")

    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--render", action="store_true", help="Render the game window.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None, help="Base seed; uses seed+episode_index.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on primitive steps per episode.")

    parser.add_argument("--beta-threshold", type=float, default=0.75, help="Switch threshold for agent mode.")
    parser.add_argument(
        "--switch-rate",
        type=int,
        default=None,
        help="If set, disable learned beta switching and force a switch every N primitive steps.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Use greedy actions and mean z instead of sampling.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.switch_rate is not None and args.switch_rate < 1:
        raise ValueError("--switch-rate must be >= 1")

    for p in [args.vlm_checkpoint, args.metacontroller_checkpoint, args.agent_checkpoint]:
        if p is not None and not os.path.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    base_model, config = load_vlm(args.vlm_checkpoint, device)
    metacontroller = load_metacontroller(args.metacontroller_checkpoint, config, device)
    control_layer = config.n_layers // 2

    agent = None
    mode_name = "train_meta-style (no agent)"
    if args.agent_checkpoint:
        agent = load_agent(args.agent_checkpoint, device)
        mode_name = "internal-rl notebook style (agent checkpoint provided)"

    render_mode = "human" if args.render else None
    env = InternalRLWrapper(device=device, seq_len=config.seq_len, render_mode=render_mode)

    print(f"Mode: {mode_name}")
    print(f"Stochastic policy: {not args.deterministic}")
    if args.switch_rate is not None:
        print(f"Switching: fixed every {args.switch_rate} steps (learned beta disabled for switching)")
    else:
        print(f"Switching: learned beta threshold ({args.beta_threshold}) in agent mode")

    returns = []
    steps_list = []
    betas = []
    switches = []

    try:
        for ep in range(args.episodes):
            ep_seed = None if args.seed is None else args.seed + ep
            if agent is None:
                stats = run_episode_train_meta_mode(
                    env=env,
                    base_model=base_model,
                    metacontroller=metacontroller,
                    config=config,
                    control_layer=control_layer,
                    max_steps=args.max_steps,
                    deterministic=args.deterministic,
                    switch_rate=args.switch_rate,
                    seed=ep_seed,
                )
                stats["switch_events"] = []
            else:
                stats = run_episode_agent_mode(
                    env=env,
                    base_model=base_model,
                    metacontroller=metacontroller,
                    agent=agent,
                    config=config,
                    control_layer=control_layer,
                    beta_threshold=args.beta_threshold,
                    max_steps=args.max_steps,
                    deterministic=args.deterministic,
                    switch_rate=args.switch_rate,
                    seed=ep_seed,
                )

            returns.append(stats["return"])
            steps_list.append(stats["steps"])
            betas.append(stats["beta_mean"])
            if stats["switches"] is not None:
                switches.append(stats["switches"])

            switch_info = ""
            if stats["switches"] is not None:
                switch_info = f" | switches={stats['switches']}"
            print(
                f"Episode {ep + 1}/{args.episodes}: "
                f"return={stats['return']:.2f} | steps={stats['steps']} | beta_mean={stats['beta_mean']:.4f}{switch_info}"
            )
            if stats["switch_events"]:
                for event in stats["switch_events"]:
                    print(
                        f"  switch#{event['switch_idx']}: frame={event['frame']} beta={event['beta']:.4f}"
                    )

        print("\nSummary")
        print(f"  return mean/std: {np.mean(returns):.3f} / {np.std(returns):.3f}")
        print(f"  return min/max: {np.min(returns):.3f} / {np.max(returns):.3f}")
        print(f"  steps mean: {np.mean(steps_list):.1f}")
        print(f"  beta mean: {np.nanmean(betas):.4f}")
        if switches:
            print(f"  switches mean: {np.mean(switches):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
