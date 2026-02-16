import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class Config:
    img_size: int = 84
    patch_size: int = 14 # 84 -> 6x6 patches.
    frame_stack: int = 4
    embed_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    seq_len: int = 64
    n_actions: int = 18
    duration_vocab_size: int = 65 # 1-64, with 0 as padding.

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch fused causal attention.
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # `attn_mask` is unused; kept for API compatibility.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MontezumaVLM(nn.Module):
    """
    Interleaved Video Transformer VLM.
    Sparse images (9) interleaved with atomic actions (64).
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.embed_dim
        self.n_patches = (config.img_size // config.patch_size) ** 2
        
        # Token embeddings.
        self.patch_embed = PatchEmbedding(
            config.img_size, config.patch_size, config.frame_stack, config.embed_dim
        )
        self.action_embed = nn.Embedding(config.n_actions + 1, config.embed_dim)
        self.start_token_id = config.n_actions
        
        # Positional and type embeddings.
        self.num_images = 9
        # 9*36 + 64 = 388, so 512 is a safe cap.
        max_tokens = 512 
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, config.embed_dim))
        self.type_embed = nn.Embedding(2, config.embed_dim) # 0 = visual, 1 = action.
        
        # Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.action_head = nn.Linear(config.embed_dim, config.n_actions)

        # Head for next-observation patch prediction.
        self.obs_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, self.n_patches * config.embed_dim)
        )

        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
    
    def forward(
        self,
        frames: torch.Tensor,      # (B, 9, 4, H, W)
        actions: torch.Tensor,     # (B, T) where T=64
        return_residuals: bool = False,
        adapter_params: tuple = None
    ) -> dict:
        B, NumImg, C, H, W = frames.shape
        T = actions.shape[1]
        device = frames.device

        # Encode visual tokens.
        frames_flat = frames.view(B * NumImg, C, H, W)
        vis_tokens = self.patch_embed(frames_flat) # (B*9, N_Patches, E)
        vis_tokens = vis_tokens.view(B, NumImg, self.n_patches, self.embed_dim)

        # Encode shifted actions.
        action_input = torch.cat([
            torch.full((B, 1), self.start_token_id, device=device, dtype=torch.long),
            actions[:, :-1]
        ], dim=1)
        act_embeds = self.action_embed(action_input) # (B, T, E)

        # Interleave visual and action tokens.
        token_list = []
        type_list = []
        action_positions = []  # Action-token positions.
        obs_pred_positions = []  # Positions used for next-observation prediction.

        num_actions = act_embeds.shape[1]
        current_pos = 0

        for i in range(NumImg):
            # Add image patch tokens.
            token_list.append(vis_tokens[:, i]) # (B, 36, E)
            type_list.append(torch.zeros(self.n_patches, dtype=torch.long, device=device))
            current_pos += self.n_patches

            # Add the matching action chunk.
            start_idx = i * 8
            end_idx = min((i + 1) * 8, num_actions)

            if start_idx < num_actions:
                chunk = act_embeds[:, start_idx : end_idx]
                token_list.append(chunk)
                type_list.append(torch.ones(chunk.shape[1], dtype=torch.long, device=device))

                # Save action-token indices.
                for j in range(chunk.shape[1]):
                    action_positions.append(current_pos + j)

                current_pos += chunk.shape[1]

                # Predict the next image after each action chunk.
                if i < NumImg - 1:
                    obs_pred_positions.append(current_pos - 1)

        tokens = torch.cat(token_list, dim=1) # (B, Total_Seq, E)

        # Add positional + type embeddings.
        seq_len_curr = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :seq_len_curr]

        types = torch.cat(type_list)
        tokens = tokens + self.type_embed(types)

        # Run transformer.
        residuals = []
        x = tokens
        for i, block in enumerate(self.blocks):
            x = block(x) # Causal attention is inside each block.

            if adapter_params is not None:
                A, B_mat, layer_idx = adapter_params
                if i == layer_idx:
                     x_uns = x.unsqueeze(-1)
                     B_T = B_mat.transpose(-1, -2)
                     term1 = torch.matmul(B_T, x_uns)
                     term2 = torch.matmul(A, term1)
                     x = x + term2.squeeze(-1)

            if return_residuals:
                residuals.append(x.clone())

        x = self.ln_f(x)

        # Predict actions at action-token positions.
        action_positions_t = torch.tensor(action_positions, device=device)
        action_embeddings = x[:, action_positions_t, :]  # (B, T, E)
        all_action_logits = self.action_head(action_embeddings)  # (B, T, n_actions)

        # Predict next-observation patches.
        if len(obs_pred_positions) > 0:
            obs_positions_t = torch.tensor(obs_pred_positions, device=device)
            obs_embeddings = x[:, obs_positions_t, :]  # (B, num_obs_preds, E)
            # Decode patch embeddings.
            obs_pred_flat = self.obs_head(obs_embeddings.reshape(-1, self.embed_dim))
            num_obs_preds = len(obs_pred_positions)
            all_obs_pred = obs_pred_flat.view(B, num_obs_preds, self.n_patches, self.embed_dim)
        else:
            all_obs_pred = None

        # Keep last-position outputs for compatibility with older code.
        final_embedding = x[:, -1, :]

        output = {
            'logits': all_action_logits[:, -1, :],  # Last-step action logits.
            'all_action_logits': all_action_logits,  # (B, T, n_actions)
            'all_obs_pred': all_obs_pred,  # (B, num_images-1, n_patches, E) or None
            'obs_pred': all_obs_pred[:, -1] if all_obs_pred is not None else None,
            'final_embedding': final_embedding,
            'action_positions': action_positions,  # Debug info.
            'obs_pred_positions': obs_pred_positions,  # Debug info.
        }
        if return_residuals:
            output['residuals'] = residuals

        return output

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Encode one frame stack into patch embeddings.

        Args:
            frame: (B, 4, H, W) frame stack

        Returns:
            (B, n_patches, embed_dim) patch embeddings
        """
        return self.patch_embed(frame)
