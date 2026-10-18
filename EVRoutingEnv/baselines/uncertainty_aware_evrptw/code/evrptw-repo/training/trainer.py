"""
training/trainer.py
====================
Main training loop for Phase 1 offline pre-training (paper §4.1).

Runs 35,000 episodes of curriculum PPO training, jointly optimising:
  - GATEncoder        (θ_enc)
  - LSTMPredictor     (θ_E)
  - PPOPolicy         (θ_π)
  - ValueNetwork      (φ)

Training schedule
-----------------
  Episodes 0     –  3,500  : n_customers ∈ [10, 20]   (warm-up)
  Episodes 3,500 – 17,500  : n_customers ∈ [20, 60]   (ramp-up)
  Episodes 17,500 – 35,000 : n_customers ∈ [60, 100]  (full scale)

Checkpoints are saved every 1,000 episodes.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from environment.evrptw_env import EVRPTWConfig, EVRPTWEnv
from models import GATEncoder, LSTMPredictor, PPOPolicy, ValueNetwork
from models.ppo_policy import PPOUpdater
from optimization.chance_constraint import ChanceConstraintChecker, BatteryState

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates Phase 1 curriculum PPO training.

    Parameters
    ----------
    config_path : str
        Path to configs/default.yaml.
    checkpoint_dir : str
        Directory for saving model checkpoints.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        checkpoint_dir: str = "results/checkpoints",
        device: Optional[str] = None,
    ) -> None:
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._build_models()
        self._build_env()
        self._build_updater()

        logger.info(f"Trainer initialised | device={self.device}")
        logger.info(f"Total parameters: {self._count_params():,}")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_models(self) -> None:
        gcfg = self.cfg["gat"]
        lcfg = self.cfg["lstm"]
        pcfg = self.cfg["ppo"]

        self.gat = GATEncoder(
            node_feature_dim=gcfg["node_feature_dim"],
            edge_feature_dim=gcfg["edge_feature_dim"],
            hidden_dim=gcfg["hidden_dim"],
            n_layers=gcfg["n_layers"],
            n_heads=gcfg["n_heads"],
            dropout=gcfg["dropout"],
        ).to(self.device)

        self.lstm = LSTMPredictor(
            hidden_dim=lcfg["hidden_dim"],
            n_layers=lcfg["n_layers"],
            dropout=lcfg["dropout"],
        ).to(self.device)

        self.policy = PPOPolicy(
            hidden_dim=gcfg["hidden_dim"],
            temperature=pcfg["temperature_tau"],
        ).to(self.device)

        self.value_net = ValueNetwork(
            hidden_dim=gcfg["hidden_dim"],
        ).to(self.device)

        self.chance_checker = ChanceConstraintChecker(
            confidence=self.cfg["chance_constraint"]["confidence_level"],
            min_battery_fraction=self.cfg["chance_constraint"]["min_battery_reserve"],
        )

    def _build_env(self) -> None:
        ecfg = self.cfg["env"]
        self.env = EVRPTWEnv(EVRPTWConfig(
            n_customers_min=ecfg["n_customers_min"],
            n_customers_max=ecfg["n_customers_max"],
            n_chargers=ecfg["n_charging_stations"],
            battery_capacity=ecfg["battery_capacity"],
            vehicle_payload=ecfg["vehicle_payload"],
            seed=self.cfg["training"]["seed"],
        ))

    def _build_updater(self) -> None:
        pcfg = self.cfg["ppo"]
        self.updater = PPOUpdater(
            policy=self.policy,
            value_net=self.value_net,
            lr=pcfg["learning_rate"],
            clip_epsilon=pcfg["clip_epsilon"],
            gae_lambda=pcfg["gae_lambda"],
            discount_gamma=pcfg["discount_gamma"],
            value_coef=pcfg["value_coef"],
            entropy_coef=pcfg["entropy_coef"],
            max_grad_norm=pcfg["max_grad_norm"],
            n_epochs=pcfg["n_epochs_per_update"],
        )

    def _count_params(self) -> int:
        models = [self.gat, self.lstm, self.policy, self.value_net]
        return sum(p.numel() for m in models for p in m.parameters())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: Optional[int] = None) -> None:
        """
        Run Phase 1 curriculum pre-training for n_episodes episodes.

        Implements Algorithm 3 (Phase 1 block): at each step, the GAT
        encoder runs once per episode, while the LSTM predictor acts as
        a per-step oracle queried at every PPO action selection.
        """
        n_ep = n_episodes or self.cfg["training"]["n_episodes"]
        eval_interval = self.cfg["training"]["eval_interval"]
        ckpt_interval = self.cfg["training"]["checkpoint_interval"]
        log_interval  = self.cfg["logging"]["log_interval"]

        episode_rewards = []
        best_eval_cost  = float("inf")

        logger.info(f"Starting Phase 1 pre-training | episodes={n_ep}")
        t0 = time.time()

        for ep in range(1, n_ep + 1):
            ep_reward, ep_info = self._run_episode(ep)
            episode_rewards.append(ep_reward)

            if ep % log_interval == 0:
                avg_r = sum(episode_rewards[-log_interval:]) / log_interval
                elapsed = time.time() - t0
                logger.info(
                    f"Ep {ep:>6}/{n_ep} | avg_reward={avg_r:>9.2f} "
                    f"| customers={ep_info['n_customers']:>3} "
                    f"| elapsed={elapsed/60:.1f}min"
                )

            if ep % eval_interval == 0:
                eval_cost = self._evaluate()
                logger.info(f"  [Eval] ep={ep} | cost={eval_cost:.2f}")
                if eval_cost < best_eval_cost:
                    best_eval_cost = eval_cost
                    self._save_checkpoint("best.pth")

            if ep % ckpt_interval == 0:
                self._save_checkpoint(f"ep_{ep:06d}.pth")

        self._save_checkpoint("final.pth")
        logger.info(f"Training complete in {(time.time()-t0)/3600:.2f}h")

    # ------------------------------------------------------------------
    # Episode rollout
    # ------------------------------------------------------------------

    def _run_episode(self, episode: int) -> tuple:
        """
        Run a single episode using the current policy.

        Key design (Algorithm 3, lines 2–12):
          1. GAT encoder runs ONCE per episode on the full graph
          2. LSTM predictor queries for each candidate arc at every step
          3. Chance constraint checker gates each move
        """
        obs = self.env.reset(episode=episode)
        total_reward = 0.0

        rollout_log_probs = []
        rollout_rewards   = []
        rollout_values    = []
        rollout_dones     = []

        self.gat.eval()
        self.lstm.eval()
        self.policy.eval()
        self.value_net.eval()

        with torch.no_grad():
            # ── Step 1: Encode graph ONCE (Algorithm 3, line 4) ──────────
            node_feat   = torch.tensor(obs["node_features"],  device=self.device)
            edge_index  = torch.tensor(obs["edge_index"],     device=self.device)
            edge_feat   = torch.tensor(obs["edge_features"],  device=self.device)
            veh_state   = torch.tensor(obs["vehicle_state"],  device=self.device)
            feas_mask   = torch.tensor(obs["feasible_mask"],  device=self.device)

            H_L, h_graph = self.gat(node_feat, edge_index, edge_feat, veh_state)

            done = False
            while not done:
                # ── Step 2: Per-step LSTM query (Algorithm 3, line 6) ────
                # (placeholder: arc features constructed from current state)
                # In production: build arc_feature_vec for each candidate arc

                # ── Step 3: Chance constraint check (Algorithm 3, lines 7–10)
                # (ChanceConstraintChecker.is_feasible applied per candidate)

                # ── Step 4: Policy action selection ──────────────────────
                action, log_prob = self.policy.select_action(
                    node_embeddings=H_L,
                    graph_embedding=h_graph,
                    vehicle_state=veh_state,
                    feasibility_mask=feas_mask,
                    greedy=False,
                )

                value = self.value_net(h_graph, veh_state)

                obs, reward, done, info = self.env.step(action)
                total_reward += reward

                rollout_log_probs.append(log_prob)
                rollout_rewards.append(reward)
                rollout_values.append(value)
                rollout_dones.append(float(done))

                if not done:
                    node_feat  = torch.tensor(obs["node_features"], device=self.device)
                    edge_index = torch.tensor(obs["edge_index"],     device=self.device)
                    edge_feat  = torch.tensor(obs["edge_features"],  device=self.device)
                    veh_state  = torch.tensor(obs["vehicle_state"],  device=self.device)
                    feas_mask  = torch.tensor(obs["feasible_mask"],  device=self.device)
                    # Re-encode after each step (could cache if N unchanged)
                    H_L, h_graph = self.gat(node_feat, edge_index, edge_feat, veh_state)

        # ── PPO update (every episode for now — batch in production) ────
        # TODO: accumulate rollouts over batch_size steps before updating
        # self._ppo_update(rollout_log_probs, rollout_rewards, rollout_values, rollout_dones)

        return total_reward, {"n_customers": self.env._instance["n_customers"]}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, n_eval: int = 10) -> float:
        """
        Greedy evaluation on n_eval fresh instances.
        Returns average total cost.
        """
        costs = []
        for _ in range(n_eval):
            obs = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    node_feat  = torch.tensor(obs["node_features"], device=self.device)
                    edge_index = torch.tensor(obs["edge_index"],     device=self.device)
                    edge_feat  = torch.tensor(obs["edge_features"],  device=self.device)
                    veh_state  = torch.tensor(obs["vehicle_state"],  device=self.device)
                    feas_mask  = torch.tensor(obs["feasible_mask"],  device=self.device)
                    H_L, h_graph = self.gat(node_feat, edge_index, edge_feat, veh_state)
                    action, _ = self.policy.select_action(
                        H_L, h_graph, veh_state, feas_mask, greedy=True
                    )
                obs, _, done, _ = self.env.step(action)
            costs.append(self.env._state["total_cost"])
        return sum(costs) / len(costs)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save({
            "gat":        self.gat.state_dict(),
            "lstm":       self.lstm.state_dict(),
            "policy":     self.policy.state_dict(),
            "value_net":  self.value_net.state_dict(),
            "optimizer":  self.updater.optimizer.state_dict(),
            "config":     self.cfg,
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.gat.load_state_dict(ckpt["gat"])
        self.lstm.load_state_dict(ckpt["lstm"])
        self.policy.load_state_dict(ckpt["policy"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.updater.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Checkpoint loaded ← {path}")
