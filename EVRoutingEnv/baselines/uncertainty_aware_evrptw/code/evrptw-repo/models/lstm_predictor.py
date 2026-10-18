"""
models/lstm_predictor.py
========================
Bidirectional LSTM probabilistic energy predictor.

Implements Algorithm 2 and Equations (3), (11) from:
  "Uncertainty-Aware Deep Reinforcement Learning for Sustainable Electric
   Vehicle Routing: A Hybrid Optimization Framework"

The predictor outputs calibrated Gaussian energy distributions (μ_E, σ²_E)
for each route arc (i, j), enabling formal chance-constrained battery
feasibility (Eq. 4 / chance constraint checker).

Training objective — Heteroscedastic negative log-likelihood (Eq. 11):
    L_hetero = (E_actual - μ_E)² / (2σ²_E) + ½ log(2π σ²_E)

Architecture
------------
  Input  : 24-dimensional arc feature vector x_ij
  BiLSTM : 2-layer bidirectional LSTM with hidden_dim = 64
  Heads  : separate linear heads for μ_E (mean) and log(σ²_E) (log-variance)
  Output : (μ_E, σ²_E)  —  σ²_E = exp( log_σ² ) to ensure positivity
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ── Feature indices within the 24-dim arc feature vector ─────────────────────
# x_ij = [d_ij, v_ij, w_ij, T_ij, grade_ij, b_i, q_rem_i, ... (17 more)]
ARC_FEATURE_NAMES = [
    "distance_km",            # 0  d_ij
    "speed_kmh",              # 1  v_ij
    "payload_kg",             # 2  w_ij
    "temperature_c",          # 3  T_ij
    "road_grade_pct",         # 4  grade_ij
    "battery_soc",            # 5  b_i / B  (state of charge at departure)
    "remaining_demand_kg",    # 6  q_rem_i
    "wind_speed_ms",          # 7
    "precipitation_mm",       # 8
    "traffic_density",        # 9
    "road_type_highway",      # 10 one-hot: highway
    "road_type_urban",        # 11 one-hot: urban
    "road_type_rural",        # 12 one-hot: rural
    "time_of_day_sin",        # 13 sin(2π t/24)
    "time_of_day_cos",        # 14 cos(2π t/24)
    "charging_flag",          # 15 1 if arc ends at charging station
    "n_customers_remaining",  # 16 normalised
    "departure_battery_kwh",  # 17 absolute battery level at departure
    "arc_length_turns",       # 18 number of turns (urban complexity proxy)
    "elevation_gain_m",       # 19
    "elevation_loss_m",       # 20
    "avg_stop_density",       # 21 stops per km
    "route_position_norm",    # 22 segment index / total segments
    "cumulative_distance_km", # 23
]
ARC_FEATURE_DIM = len(ARC_FEATURE_NAMES)  # 24


class LSTMPredictor(nn.Module):
    """
    Bidirectional LSTM probabilistic energy predictor (Algorithm 2).

    Parameters
    ----------
    input_dim : int
        Arc feature dimension (24).
    hidden_dim : int
        LSTM hidden state dimension per direction (64).
    n_layers : int
        Number of BiLSTM layers (2).
    dropout : float
        Dropout probability between LSTM layers.
    """

    def __init__(
        self,
        input_dim: int = ARC_FEATURE_DIM,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input normalisation
        self.input_norm = nn.LayerNorm(input_dim)

        # Bidirectional LSTM (Alg. 2, line 5)
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Concatenated hidden dim: 2 × hidden_dim (forward + backward)
        concat_dim = 2 * hidden_dim

        # Intermediate projection
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate output heads (Alg. 2, line 7)
        self.head_mean = nn.Linear(concat_dim, 1)       # W_μ → μ_E
        self.head_logvar = nn.Linear(concat_dim, 1)     # W_σ → log σ²_E

        # Clamp bounds to avoid numerical instability
        self._log_var_min = -6.0   # σ² ≥ exp(-6) ≈ 0.0025 kWh²
        self._log_var_max = 4.0    # σ² ≤ exp(4)  ≈ 54.6   kWh²

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        arc_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for a batch of route sequences.

        Parameters
        ----------
        arc_features : Tensor, shape (B, T, 24)
            Batch of B routes, each with T arc feature vectors.
            For single-arc inference (online routing), use shape (1, 1, 24).

        Returns
        -------
        mu_E : Tensor, shape (B, T)
            Predicted mean energy consumption in kWh per arc.
        sigma2_E : Tensor, shape (B, T)
            Predicted energy variance σ²_E in kWh².
        """
        # Input normalisation
        x = self.input_norm(arc_features)            # (B, T, 24)

        # BiLSTM (Alg. 2, lines 5–6)
        h, _ = self.bilstm(x)                        # (B, T, 2·hidden_dim)
        h = self.proj(h)

        # Mean prediction (Alg. 2, line 7a)
        mu_E = self.head_mean(h).squeeze(-1)         # (B, T)  — unbounded

        # Log-variance prediction with clamping (Alg. 2, line 7b)
        log_var = self.head_logvar(h).squeeze(-1)    # (B, T)
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)
        sigma2_E = torch.exp(log_var)                # (B, T)  — always > 0

        return mu_E, sigma2_E

    @property
    def log_var_min(self) -> float:
        return self._log_var_min

    @property
    def log_var_max(self) -> float:
        return self._log_var_max

    def predict_single_arc(
        self,
        arc_feature_vec: Tensor,
    ) -> Tuple[float, float]:
        """
        Convenience method for per-step online inference (Algorithm 3, line 6).

        Parameters
        ----------
        arc_feature_vec : Tensor, shape (24,)
            Feature vector for a single candidate arc (i, j).

        Returns
        -------
        mu_E : float
            Predicted mean energy consumption (kWh).
        sigma2_E : float
            Predicted energy variance (kWh²).
        """
        self.eval()
        with torch.no_grad():
            x = arc_feature_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, 24)
            mu, var = self.forward(x)
        return mu.item(), var.item()


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic negative log-likelihood loss for LSTM training (Eq. 11).

        L_hetero = (E_actual - μ_E)² / (2 σ²_E) + ½ log(2π σ²_E)

    The two terms trade off prediction accuracy against uncertainty calibration:
    predicting overly wide intervals is penalised by the log term.
    """

    def forward(
        self,
        mu_E: Tensor,
        sigma2_E: Tensor,
        E_actual: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        mu_E : Tensor
            Predicted mean energy (any shape).
        sigma2_E : Tensor
            Predicted variance (same shape as mu_E, strictly positive).
        E_actual : Tensor
            Ground-truth energy consumption (same shape).

        Returns
        -------
        Tensor : scalar loss value (mean over batch).
        """
        nll = (E_actual - mu_E) ** 2 / (2.0 * sigma2_E) \
              + 0.5 * torch.log(2.0 * torch.pi * sigma2_E)
        return nll.mean()


def coverage_rate(
    mu_E: Tensor,
    sigma2_E: Tensor,
    E_actual: Tensor,
    confidence: float = 0.95,
) -> float:
    """
    Compute empirical coverage rate of the predicted confidence interval.

    Expected to match ``confidence`` (e.g., 94.7% ≈ 95%) for a calibrated
    predictor.  Used as the primary calibration metric in Section 5.4.

    Parameters
    ----------
    mu_E, sigma2_E, E_actual : Tensor
        Predictions and ground truth.
    confidence : float
        Target coverage level (0.95 → 95% CI).

    Returns
    -------
    float : fraction of samples where E_actual falls inside the predicted CI.
    """
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)          # 1.96 for 95%
    sigma_E = sigma2_E.sqrt()
    lower = mu_E - z * sigma_E
    upper = mu_E + z * sigma_E
    inside = ((E_actual >= lower) & (E_actual <= upper)).float()
    return inside.mean().item()
