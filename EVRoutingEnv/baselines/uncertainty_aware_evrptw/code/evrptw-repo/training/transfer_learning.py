"""
training/transfer_learning.py
==============================
Three-stage transfer learning protocol for domain adaptation (paper §4.4).

Reduces full re-training time (72 h) to 3–5 h using 400–600 local routes.

Stage 1 — LSTM Recalibration
    Fine-tune θ_E on local route energy data (heteroscedastic loss, Eq. 11).
    GAT encoder and PPO policy remain frozen.

Stage 2 — Policy Adaptation
    Fine-tune θ_π via behavioural cloning + PPO (Eq. 14):
        L_adapt = λ_BC · (-Σ log π_{θ_π}(a*_t|s_t)) + λ_RL · L^CLIP(θ_π)
    GAT encoder stays frozen.

Stage 3 — Uncertainty Recalibration
    Validate empirical coverage on held-out local routes.
    Apply temperature scaling to LSTM variance output if coverage drifts.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from models import GATEncoder, LSTMPredictor, PPOPolicy, ValueNetwork
from models.lstm_predictor import HeteroscedasticLoss, coverage_rate

logger = logging.getLogger(__name__)


class TransferLearner:
    """
    Three-stage transfer learning for deploying to a new operational region.

    Parameters
    ----------
    pretrained_checkpoint : str
        Path to Phase 1 pre-trained checkpoint (.pth).
    config_path : str
        Path to configs/transfer.yaml.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        config_path: str = "configs/transfer.yaml",
        device: Optional[str] = None,
    ) -> None:
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._load_pretrained(pretrained_checkpoint)
        logger.info(f"TransferLearner ready | device={self.device}")

    # ------------------------------------------------------------------
    # Load pretrained checkpoint
    # ------------------------------------------------------------------

    def _load_pretrained(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt["config"]
        gcfg = cfg["gat"]
        lcfg = cfg["lstm"]
        pcfg = cfg["ppo"]

        self.gat = GATEncoder(
            node_feature_dim=gcfg["node_feature_dim"],
            edge_feature_dim=gcfg["edge_feature_dim"],
            hidden_dim=gcfg["hidden_dim"],
            n_layers=gcfg["n_layers"],
            n_heads=gcfg["n_heads"],
        ).to(self.device)
        self.gat.load_state_dict(ckpt["gat"])

        self.lstm = LSTMPredictor(
            hidden_dim=lcfg["hidden_dim"],
            n_layers=lcfg["n_layers"],
        ).to(self.device)
        self.lstm.load_state_dict(ckpt["lstm"])

        self.policy = PPOPolicy(
            hidden_dim=gcfg["hidden_dim"],
            temperature=pcfg["temperature_tau"],
        ).to(self.device)
        self.policy.load_state_dict(ckpt["policy"])

        self.value_net = ValueNetwork(
            hidden_dim=gcfg["hidden_dim"],
        ).to(self.device)
        self.value_net.load_state_dict(ckpt["value_net"])

        logger.info(f"Pre-trained weights loaded from {checkpoint_path}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        local_route_data: "dict",
        output_dir: str = "results/transfer",
        target_coverage: float = 0.95,
    ) -> dict:
        """
        Execute all three transfer learning stages.

        Parameters
        ----------
        local_route_data : dict with keys:
            "train"     — list of (arc_features, energy_actual) tuples
            "val"       — same format, held-out validation set
            "routes"    — list of expert routes for behavioural cloning
        output_dir : str
            Directory to save adapted checkpoint.
        target_coverage : float
            Target empirical coverage (0.95).

        Returns
        -------
        dict : summary metrics for each stage
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = {}

        logger.info("=== Stage 1: LSTM Energy Predictor Calibration ===")
        results["stage1"] = self._stage1_lstm_calibration(local_route_data)

        logger.info("=== Stage 2: Policy Adaptation ===")
        results["stage2"] = self._stage2_policy_adaptation(local_route_data)

        logger.info("=== Stage 3: Uncertainty Recalibration ===")
        results["stage3"] = self._stage3_recalibration(
            local_route_data, target_coverage
        )

        # Save adapted checkpoint
        out_path = Path(output_dir) / "adapted.pth"
        torch.save({
            "gat":       self.gat.state_dict(),
            "lstm":      self.lstm.state_dict(),
            "policy":    self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
        }, out_path)
        logger.info(f"Adapted checkpoint saved → {out_path}")
        return results

    # ------------------------------------------------------------------
    # Stage 1: LSTM fine-tuning on local energy data
    # ------------------------------------------------------------------

    def _stage1_lstm_calibration(self, data: dict) -> dict:
        """
        Fine-tune only θ_E on 400–600 local route segments.
        GAT encoder and policy network are frozen throughout.
        """
        s1cfg = self.cfg["stage1_lstm_calibration"]

        # Freeze everything except LSTM
        self._freeze(self.gat)
        self._freeze(self.policy)
        self._freeze(self.value_net)
        self._unfreeze(self.lstm)

        optimizer = torch.optim.AdamW(
            self.lstm.parameters(), lr=s1cfg["learning_rate"]
        )
        criterion = HeteroscedasticLoss()

        train_data = data["train"]
        val_data   = data["val"]

        best_val_mape = float("inf")
        patience = s1cfg["early_stopping_patience"]
        stagnation = 0

        logger.info(
            f"  Fine-tuning LSTM on {len(train_data)} local segments "
            f"for up to {s1cfg['n_epochs']} epochs"
        )

        for epoch in range(1, s1cfg["n_epochs"] + 1):
            self.lstm.train()
            train_loss = 0.0

            for arc_feat, e_actual in train_data:
                x = torch.tensor(arc_feat, device=self.device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor([[e_actual]], device=self.device)

                mu_E, sigma2_E = self.lstm(x)
                loss = criterion(mu_E, sigma2_E, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.lstm.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            val_mape, val_cov = self._eval_lstm(val_data)
            train_loss /= max(len(train_data), 1)

            if epoch % 10 == 0:
                logger.info(
                    f"  Epoch {epoch:>3} | train_loss={train_loss:.4f} "
                    f"| val_MAPE={val_mape:.2%} | val_coverage={val_cov:.3f}"
                )

            if val_mape < best_val_mape - 1e-4:
                best_val_mape = val_mape
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        final_mape, final_cov = self._eval_lstm(val_data)
        logger.info(
            f"  Stage 1 complete | final MAPE={final_mape:.2%} "
            f"| coverage={final_cov:.3f} (target 0.95)"
        )
        return {"mape": final_mape, "coverage": final_cov}

    def _eval_lstm(self, val_data: list) -> tuple:
        self.lstm.eval()
        errors, mu_all, var_all, y_all = [], [], [], []
        with torch.no_grad():
            for arc_feat, e_actual in val_data:
                x = torch.tensor(arc_feat, device=self.device).unsqueeze(0).unsqueeze(0)
                mu_E, sigma2_E = self.lstm(x)
                mu_all.append(mu_E.squeeze())
                var_all.append(sigma2_E.squeeze())
                y_all.append(torch.tensor(e_actual, device=self.device))
                errors.append(abs(mu_E.item() - e_actual) / (e_actual + 1e-8))

        mu_t  = torch.stack(mu_all)
        var_t = torch.stack(var_all)
        y_t   = torch.stack(y_all)
        mape = sum(errors) / len(errors)
        cov  = coverage_rate(mu_t, var_t, y_t)
        return mape, cov

    # ------------------------------------------------------------------
    # Stage 2: Policy adaptation via BC + PPO
    # ------------------------------------------------------------------

    def _stage2_policy_adaptation(self, data: dict) -> dict:
        """
        Fine-tune θ_π using behavioural cloning loss + PPO (Eq. 14).
        GAT encoder remains frozen; LSTM is now calibrated.
        """
        s2cfg = self.cfg["stage2_policy_adaptation"]
        bc_lam = s2cfg["bc_lambda"]
        rl_lam = s2cfg["rl_lambda"]

        self._freeze(self.gat)
        self._unfreeze(self.policy)
        # LSTM frozen after Stage 1 calibration
        self._freeze(self.lstm)

        optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=s2cfg["learning_rate"]
        )

        expert_routes = data.get("routes", [])
        n_episodes = s2cfg["n_episodes"]
        bc_losses, rl_losses = [], []

        logger.info(
            f"  Policy adaptation | BC episodes={len(expert_routes)} "
            f"| RL episodes={n_episodes}"
        )

        # Behavioural cloning from expert routes
        if expert_routes:
            self.policy.train()
            for route_data in expert_routes:
                bc_loss = self._behavioural_cloning_step(route_data, optimizer, bc_lam)
                bc_losses.append(bc_loss)

        # RL fine-tuning episodes (placeholder — wire in full PPO rollouts)
        # TODO: integrate with EVRPTWEnv + PPOUpdater for n_episodes
        logger.info(
            f"  Stage 2 complete | avg_BC_loss={sum(bc_losses)/max(len(bc_losses),1):.4f}"
        )
        return {"bc_loss": sum(bc_losses) / max(len(bc_losses), 1)}

    def _behavioural_cloning_step(
        self, route_data: dict, optimizer, lambda_bc: float
    ) -> float:
        """Single BC gradient step: maximise log-probability of expert actions."""
        # route_data: {"node_features", "edge_index", "edge_features",
        #              "vehicle_states", "expert_actions"}
        self.policy.train()
        optimizer.zero_grad()

        # TODO: encode graph through GAT, compute log-probs of expert actions
        # Placeholder return
        return 0.0

    # ------------------------------------------------------------------
    # Stage 3: Temperature scaling for uncertainty recalibration
    # ------------------------------------------------------------------

    def _stage3_recalibration(
        self, data: dict, target_coverage: float
    ) -> dict:
        """
        Validate empirical coverage on 100 held-out local routes.
        If coverage deviates from target, apply temperature scaling to σ²_E.
        """
        val_data = data.get("val", [])[:100]
        _, current_cov = self._eval_lstm(val_data)

        logger.info(
            f"  Coverage before scaling: {current_cov:.3f} "
            f"(target {target_coverage})"
        )

        if abs(current_cov - target_coverage) > 0.01:
            # Binary search for temperature T such that σ²_scaled = T² · σ²_E
            # achieves target empirical coverage
            T = self._find_temperature(val_data, target_coverage)
            logger.info(f"  Temperature scaling applied: T={T:.4f}")
            # TODO: store T and apply at inference time in ChanceConstraintChecker
        else:
            T = 1.0
            logger.info("  No temperature scaling needed — coverage within tolerance")

        _, recal_cov = self._eval_lstm(val_data)
        logger.info(f"  Coverage after scaling: {recal_cov:.3f}")
        return {"temperature": T, "coverage_before": current_cov,
                "coverage_after": recal_cov}

    def _find_temperature(
        self, val_data: list, target_cov: float, n_iter: int = 20
    ) -> float:
        """Binary search for optimal variance temperature T ∈ [0.5, 3.0]."""
        lo, hi = 0.5, 3.0
        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            # TODO: apply T to σ²_E and recompute coverage
            cov = target_cov   # placeholder
            if cov < target_cov:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    @staticmethod
    def _unfreeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = True
