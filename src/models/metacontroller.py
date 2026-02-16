import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LRU(nn.Module):
    """
    Linear Recurrent Unit (LRU), the core Hawk recurrence.

    Implements: x_t = λ * x_{t-1} + (1 - λ) * (W_in * u_t)
                y_t = W_out * x_t

    λ is a learnable diagonal decay.
    """
    def __init__(self, d_input, d_state, n_heads=8):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.n_heads = n_heads
        self.head_dim = d_state // n_heads

        # Input projection.
        self.W_in = nn.Linear(d_input, d_state, bias=False)

        # Learnable decay λ in (0, 1), initialized near 0.9.
        self.log_lambda = nn.Parameter(torch.ones(d_state) * 2.0)

        # Output projection.
        self.W_out = nn.Linear(d_state, d_input, bias=False)

        # Learned skip term.
        self.D = nn.Parameter(torch.ones(d_input) * 0.1)

    def forward(self, u, reverse=False):
        """
        Args:
            u: (B, T, d_input)
            reverse: process sequence in reverse
        Returns:
            y: (B, T, d_input)
        """
        B, T, _ = u.shape
        device = u.device

        if reverse:
            u = torch.flip(u, dims=[1])

        # Compute decay.
        lam = torch.sigmoid(self.log_lambda)  # (d_state,)

        # Project input.
        u_proj = self.W_in(u)  # (B, T, d_state)

        # Run recurrence.
        x = torch.zeros(B, self.d_state, device=device)
        ys = []

        for t in range(T):
            # Recurrent update.
            x = lam * x + (1 - lam) * u_proj[:, t]
            ys.append(x)

        # Stack hidden states.
        y_states = torch.stack(ys, dim=1)  # (B, T, d_state)

        # Project back with skip.
        y = self.W_out(y_states) + u * self.D

        if reverse:
            y = torch.flip(y, dims=[1])

        return y

    def step(self, u_t, state=None):
        """Single recurrent step.
        Args:
            u_t: (B, d_input) single timestep input
            state: (B, d_state) previous hidden state or None
        Returns:
            y_t: (B, d_input) output
            new_state: (B, d_state) updated state
        """
        lam = torch.sigmoid(self.log_lambda)
        u_proj = self.W_in(u_t)  # (B, d_state)

        if state is None:
            state = torch.zeros_like(u_proj)

        new_state = lam * state + (1 - lam) * u_proj
        y_t = self.W_out(new_state) + u_t * self.D
        return y_t, new_state


class HawkBlock(nn.Module):
    """
    Hawk block with pre-norm:
    LRU mixing + MLP mixing, both with residuals.
    """
    def __init__(self, d_model, d_state=256, n_heads=8, mlp_ratio=2, dropout=0.0):
        super().__init__()

        # Sequence mixing.
        self.norm1 = nn.LayerNorm(d_model)
        self.lru = LRU(d_model, d_state, n_heads)

        # Channel mixing.
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.ReLU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, reverse=False):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model)
        """
        # Sequence mixing with residual.
        x = x + self.dropout(self.lru(self.norm1(x), reverse=reverse))

        # Channel mixing with residual.
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x

    def step(self, x_t, state=None):
        """Single recurrent step.
        Args:
            x_t: (B, d_model)
            state: (B, d_state) LRU state or None
        Returns:
            out: (B, d_model)
            new_state: (B, d_state)
        """
        residual = x_t
        lru_out, new_state = self.lru.step(self.norm1(x_t), state)
        x_t = residual + lru_out
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, new_state


class BidirectionalHawk(nn.Module):
    """
    Bidirectional Hawk encoder for sequence embeddings.
    """
    def __init__(self, d_input, d_output, d_state=256, n_heads=8, n_layers=1, dropout=0.0):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        # Optional input projection.
        self.input_proj = nn.Linear(d_input, d_output) if d_input != d_output else nn.Identity()

        # Forward stack.
        self.forward_layers = nn.ModuleList([
            HawkBlock(d_output, d_state, n_heads, mlp_ratio=2, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Backward stack.
        self.backward_layers = nn.ModuleList([
            HawkBlock(d_output, d_state, n_heads, mlp_ratio=2, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Merge forward/backward summaries.
        self.output_proj = nn.Linear(d_output * 2, d_output)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_input) input sequence
        Returns:
            embed: (B, d_output) sequence embedding
        """
        # Project input.
        x = self.input_proj(x)  # (B, T, d_output)

        # Forward pass.
        h_fwd = x
        for layer in self.forward_layers:
            h_fwd = layer(h_fwd, reverse=False)

        # Backward pass.
        h_bwd = x
        for layer in self.backward_layers:
            h_bwd = layer(h_bwd, reverse=True)

        # Pool final states from both directions.
        fwd_final = h_fwd[:, -1, :]  # (B, d_output) - end of forward
        bwd_final = h_bwd[:, 0, :]   # (B, d_output) - start of backward (saw whole seq)

        # Concatenate and project.
        combined = torch.cat([fwd_final, bwd_final], dim=-1)  # (B, d_output*2)
        embed = self.output_proj(combined)  # (B, d_output)

        return embed


class Metacontroller(nn.Module):
    """
    Metacontroller module.

    1. h_t = GRU(e_t, h_{t-1})  -- history summary (only e_t as input)
    2. s = f_emb(e_{1:T})       -- sequence embedding
    3. μ_t, Σ_t = f_enc(e_t, h_{t-1}, s)  -- encoder
    4. β_t = f_switch(e_t, h_{t-1}, z_{t-1})  -- switching unit
    5. z_t = β_t * z̃_t + (1-β_t) * z_{t-1}  -- temporal integration
    6. U_t = f_hyp(z_t)  -- hypernetwork decoder

    Optional auxiliary supervision:
    - Position head: z -> (x, y)

    Waypoint-conditioned mode:
    - β_t = 1 forced (no temporal integration)
    - z_t = waypoint_embedding[waypoint_idx] (ground truth injection)
    - Skips VAE encoder entirely
    """

    def __init__(self, config, base_embed_dim, aux_position_predictor=False,
                 waypoint_conditioned=False, num_waypoints=6):
        super().__init__()
        self.config = config
        self.base_embed_dim = base_embed_dim
        self.latent_dim = 8      # Latent size.
        self.rank = 16           # Low-rank adapter
        self.n_h = 32            # History hidden size.
        self.n_s = 32            # Sequence embedding size.
        self.aux_position_predictor = aux_position_predictor
        self.waypoint_conditioned = waypoint_conditioned
        self.num_waypoints = num_waypoints

        # 1) History GRU.
        self.history_gru = nn.GRUCell(base_embed_dim, self.n_h)

        # 2) Sequence embedder.
        self.seq_embedder = BidirectionalHawk(
            d_input=base_embed_dim,
            d_output=self.n_s,
            d_state=64,      # Smaller state size for the metacontroller.
            n_heads=4,
            n_layers=1,
            dropout=0.0
        )

        # 3) Encoder.
        self.encoder = nn.Sequential(
            nn.Linear(base_embed_dim + self.n_h + self.n_s, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.latent_dim)  # mu and logvar.
        )

        # 4) Switching unit.
        self.switch_net = nn.Sequential(
            nn.Linear(base_embed_dim + self.n_h + self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Start with a mild bias toward switching.
        with torch.no_grad():
            self.switch_net[-1].bias.fill_(1.0)

        # 6) Decoder hypernetwork.
        self.hypernet = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * base_embed_dim * self.rank)  # A/B adapter matrices.
        )

        # Optional position predictor from z.
        if aux_position_predictor:
            self.position_predictor = nn.Sequential(
                nn.Linear(self.latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # Predict normalized (x, y).
            )

        # Optional waypoint conditioning.
        if waypoint_conditioned:
            # Learned waypoint embeddings.
            self.waypoint_embedding = nn.Embedding(num_waypoints, self.latent_dim)

    def forward(self, residuals, training=True, waypoint_idx=None,
                waypoint_indices=None, waypoint_switches=None, seq_context=None,
                fixed_switch_rate=None):
        """
        Forward pass through metacontroller.

        Args:
            residuals: (B, T, D) - residual stream activations e_{1:T,l}
            training: bool - if True, use sequence embedding
            waypoint_idx: (B,) - static waypoint indices (same for all timesteps)
            waypoint_indices: (B, T) - per-timestep waypoint targets
            waypoint_switches: (B, T) - binary indicators when waypoint changed
            seq_context: optional clean residuals for sequence embedding
            fixed_switch_rate: if set, force a switch every N steps

        Returns:
            dict with: mus, logvars, betas, zs, us
        """
        B, T, D = residuals.shape
        device = residuals.device

        # Waypoint-conditioned mode.
        if self.waypoint_conditioned and (waypoint_idx is not None or waypoint_indices is not None):
            return self._forward_waypoint_conditioned(
                residuals, waypoint_idx=waypoint_idx,
                waypoint_indices=waypoint_indices, waypoint_switches=waypoint_switches
            )

        # Standard VAE-style path.
        # 2) Sequence embedding (training only).
        if training:
            # Use seq_context when provided; otherwise use residuals.
            context_for_seq = seq_context if seq_context is not None else residuals
            # Returns (B, n_s).
            seq_embed = self.seq_embedder(context_for_seq)
        else:
            # No future context at test time.
            seq_embed = torch.zeros(B, self.n_s, device=device)

        # Initial recurrent states.
        h_prev = torch.zeros(B, self.n_h, device=device)  # h_{t-1}
        z_prev = torch.zeros(B, self.latent_dim, device=device)  # z_{t-1}

        # Outputs.
        mus, logvars = [], []
        betas = []
        beta_logits = []
        zs = []
        us = []
        kls = []

        for t in range(T):
            e_t = residuals[:, t]  # Current residual (B, D)

            # 4) Switching unit (uses previous h and z).
            switch_input = torch.cat([e_t, h_prev, z_prev], dim=1)
            beta_logit_t = self.switch_net(switch_input)  # (B, 1) logits.
            beta_t = torch.sigmoid(beta_logit_t)  # (B, 1) probabilities.
            beta_logits.append(beta_logit_t)

            # Optional fixed switch schedule.
            if fixed_switch_rate is not None and fixed_switch_rate > 0:
                if t % fixed_switch_rate == 0:
                    beta_t = torch.ones_like(beta_t)
                else:
                    beta_t = torch.zeros_like(beta_t)

            betas.append(beta_t)

            # 3) Encoder (uses previous h).
            enc_input = torch.cat([e_t, h_prev, seq_embed], dim=1)
            enc_out = self.encoder(enc_input)  # (B, 2*latent_dim)
            mu_t = enc_out[:, :self.latent_dim]
            logvar_t = enc_out[:, self.latent_dim:]
            mus.append(mu_t)
            logvars.append(logvar_t)

            # Sample latent and compute KL terms.
            z_t, mq, mp, sq, sp = self._beta_conditional_sample(mu_t, logvar_t, z_prev, beta_t)
            kl_t = 0.5 * torch.sum(
                torch.log(sp + 1e-8) - torch.log(sq + 1e-8)
                + (sq + (mq - mp).pow(2)) / (sp + 1e-8)
                - 1.0,
                dim=-1
            )
            kls.append(kl_t)
            zs.append(z_t)

            # 6) Decode adapter params.
            params = self.hypernet(z_t)  # (B, 2*D*R)
            params = params.view(B, 2, self.base_embed_dim, self.rank)
            A_t = params[:, 0]  # (B, D, R)
            B_t = params[:, 1]  # (B, D, R)
            us.append((A_t, B_t))

            # 1) Update history after using h_{t-1}.
            h_t = self.history_gru(e_t, h_prev)

            # Advance state.
            h_prev = h_t
            z_prev = z_t.detach() if training else z_t

        result = {
            'mus': torch.stack(mus, dim=1),       # (B, T, latent_dim)
            'logvars': torch.stack(logvars, dim=1),  # (B, T, latent_dim)
            'betas': torch.stack(betas, dim=1),   # (B, T, 1)
            'beta_logits': torch.stack(beta_logits, dim=1),  # (B, T, 1), useful for BCE.
            'zs': torch.stack(zs, dim=1),         # (B, T, latent_dim)
            'kl': torch.stack(kls, dim=1),        # (B, T)
            'us': us,  # List of T tuples (A_t, B_t), each (B, D, R).
            'us_stacked': (
                torch.stack([u[0] for u in us], dim=1),
                torch.stack([u[1] for u in us], dim=1),
            )
        }

        # Optional position predictions.
        if self.aux_position_predictor:
            zs_tensor = result['zs']  # (B, T, latent_dim)
            position_preds = self.position_predictor(zs_tensor)  # (B, T, 2)
            result['position_preds'] = position_preds

        return result

    def _forward_waypoint_conditioned(self, residuals, waypoint_idx=None,
                                       waypoint_indices=None, waypoint_switches=None):
        """
        Waypoint-conditioned forward pass with temporal integration.

        Supports two modes:
        1) Static: one waypoint for all steps.
        2) Dynamic: per-step waypoints with switch markers.

        Args:
            residuals: (B, T, D) - residual stream activations
            waypoint_idx: (B,) - static waypoint index
            waypoint_indices: (B, T) - per-timestep waypoint targets
            waypoint_switches: (B, T) - binary indicators of when waypoint changed

        Returns:
            dict with: zs, us, us_stacked, betas
        """
        B, T, D = residuals.shape
        device = residuals.device

        use_dynamic = waypoint_indices is not None

        if use_dynamic:
            # Dynamic mode with temporal integration.
            T_data = waypoint_indices.shape[1]  # Original data length (64)

            # Resample waypoint annotations if lengths differ.
            if T_data != T:
                waypoint_indices = F.interpolate(
                    waypoint_indices.float().unsqueeze(1),
                    size=T, mode='nearest'
                ).squeeze(1).long()

                if waypoint_switches is not None:
                    waypoint_switches = F.interpolate(
                        waypoint_switches.unsqueeze(1),
                        size=T, mode='nearest'
                    ).squeeze(1)

            all_embeddings = self.waypoint_embedding.weight  # (num_waypoints, latent_dim)

            # Target embedding per timestep.
            target_embeddings = all_embeddings[waypoint_indices]  # (B, T, latent_dim)

            # Beta is 1 at t=0 and at switch points.
            if waypoint_switches is not None:
                betas = waypoint_switches.clone().unsqueeze(-1)  # (B, T, 1)
            else:
                betas = torch.zeros(B, T, 1, device=device)
            betas[:, 0, :] = 1.0  # Always switch at t=0

            # Temporal integration for z.
            zs_list = []
            z_prev = torch.zeros(B, self.latent_dim, device=device)

            for t in range(T):
                beta_t = betas[:, t, :]  # (B, 1)
                target_t = target_embeddings[:, t, :]  # (B, latent_dim)
                z_t = beta_t * target_t + (1 - beta_t) * z_prev
                zs_list.append(z_t)
                z_prev = z_t

            zs = torch.stack(zs_list, dim=1)  # (B, T, latent_dim)

            # Decode adapters for each timestep.
            zs_flat = zs.view(B * T, self.latent_dim)
            params_flat = self.hypernet(zs_flat)  # (B*T, 2*D*R)
            params = params_flat.view(B, T, 2, self.base_embed_dim, self.rank)
            A_expanded = params[:, :, 0, :, :]  # (B, T, D, R)
            B_expanded = params[:, :, 1, :, :]  # (B, T, D, R)

        else:
            # Static mode: one waypoint for all timesteps.
            z = self.waypoint_embedding(waypoint_idx)  # (B, latent_dim)

            # Decode adapters.
            params = self.hypernet(z)  # (B, 2*D*R)
            params = params.view(B, 2, self.base_embed_dim, self.rank)
            A = params[:, 0]  # (B, D, R)
            B_mat = params[:, 1]  # (B, D, R)

            # Repeat across timesteps.
            zs = z.unsqueeze(1).expand(-1, T, -1).contiguous()  # (B, T, latent_dim)
            betas = torch.ones(B, T, 1, device=device)  # β = 1 always

            # Expand adapters to (B, T, D, R).
            A_expanded = A.unsqueeze(1).expand(-1, T, -1, -1).contiguous()
            B_expanded = B_mat.unsqueeze(1).expand(-1, T, -1, -1).contiguous()

        # Keep list format for older call sites.
        us = [(A_expanded[:, t].contiguous(), B_expanded[:, t].contiguous()) for t in range(T)]

        result = {
            'zs': zs,
            'betas': betas,
            'us': us,
            'us_stacked': (A_expanded.contiguous(), B_expanded.contiguous()),
            'mus': None,
            'logvars': None,
            'kl': torch.zeros(B, T, device=device),
        }

        # Optional position predictions.
        if self.aux_position_predictor:
            position_preds = self.position_predictor(zs)  # (B, T, 2)
            result['position_preds'] = position_preds

        return result

    def step(self, e_t, h_prev, z_prev, seq_embed=None):
        """
        Single step for online/RL use.

        Args:
            e_t: (B, D) current residual
            h_prev: (B, n_h) previous hidden state
            z_prev: (B, latent_dim) previous latent
            seq_embed: (B, n_s) sequence embedding (zeros at test time)

        Returns:
            z_t, h_t, beta_t, (A_t, B_t)
        """
        B = e_t.shape[0]
        device = e_t.device

        if seq_embed is None:
            seq_embed = torch.zeros(B, self.n_s, device=device)

        # Switching step.
        switch_input = torch.cat([e_t, h_prev, z_prev], dim=1)
        beta_t = torch.sigmoid(self.switch_net(switch_input))

        # Encoder step.
        enc_input = torch.cat([e_t, h_prev, seq_embed], dim=1)
        enc_out = self.encoder(enc_input)
        mu_t = enc_out[:, :self.latent_dim]
        logvar_t = enc_out[:, self.latent_dim:]

        # Sample latent.
        z_t, _, _, _, _ = self._beta_conditional_sample(mu_t, logvar_t, z_prev, beta_t)

        # Decoder step.
        params = self.hypernet(z_t)
        params = params.view(B, 2, self.base_embed_dim, self.rank)
        A_t = params[:, 0]
        B_t = params[:, 1]

        # Update recurrent state.
        h_t = self.history_gru(e_t, h_prev)

        return z_t, h_t, beta_t, (A_t, B_t)

    @staticmethod
    def _beta_conditional_sample(mu_t, logvar_t, z_prev, beta_t, beta_eps=1e-3):
        """
        Sample beta-conditional z and return KL stats.
        """
        beta = beta_t.clamp(0.0, 1.0)
        beta_eff = torch.maximum(
            beta,
            torch.tensor(beta_eps, device=beta.device, dtype=beta.dtype)
        )

        sigma = torch.exp(0.5 * logvar_t)
        mq = (1.0 - beta) * z_prev + beta * mu_t
        sq = (beta_eff * sigma).pow(2)

        mp = (1.0 - beta) * z_prev
        sp = beta_eff.pow(2)
        if sp.shape[-1] == 1:
            sp = sp.expand_as(mu_t)

        eps = torch.randn_like(mu_t)
        z_t = mq + torch.sqrt(sq + 1e-8) * eps
        return z_t, mq, mp, sq, sp

    def step_with_z(self, e_t, z, h_switch_prev):
        """
        Single RL step with externally provided z.

        Args:
            e_t: (B, D) current residual embedding
            z: (B, latent_dim) latent action from policy
            h_switch_prev: (B, n_h) previous switch hidden state

        Returns:
            beta_t: (B, 1) switching probability
            h_switch_t: (B, n_h) updated hidden state
            (A_t, B_t): adapter matrices
        """
        B = e_t.shape[0]

        # Switching step.
        switch_input = torch.cat([e_t, h_switch_prev, z], dim=1)
        beta_t = torch.sigmoid(self.switch_net(switch_input))

        # Update recurrent state.
        h_switch_t = self.history_gru(e_t, h_switch_prev)

        # Decode adapter params.
        params = self.hypernet(z)
        params = params.view(B, 2, self.base_embed_dim, self.rank)
        A_t = params[:, 0]  # (B, D, R)
        B_t = params[:, 1]  # (B, D, R)

        return beta_t, h_switch_t, (A_t, B_t)
