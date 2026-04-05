"""
eval_agent.py — Evaluate a trained IDQN policy for WarehouseMultiEnv

Runs N evaluation episodes with no exploration (ε=0) and prints a full
performance report: scores, delivery rate, steps efficiency, and per-robot stats.

Usage:
  python eval_agent.py --model checkpoints/best_policy.pt
  python eval_agent.py --model checkpoints/best_policy.pt --episodes 20 --render
  python eval_agent.py --model checkpoints/ckpt_ep01000.pt --episodes 50 --no-render
"""

import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from warehouse_multi_env import WarehouseMultiEnv, NUM_AGENTS, OBS_SIZE, N_ACTIONS

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Network (must match trainer exactly) ─────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),      nn.LayerNorm(256), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V    = self.value_stream(feat)
        A    = self.advantage_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))


# ── Evaluator ─────────────────────────────────────────────────────────────────
class AgentEvaluator:
    def __init__(self, model_path: str, render: bool = False):
        self.render  = render
        self.agents  = [f"robot_{i}" for i in range(NUM_AGENTS)]

        # Load policy
        self.policy = DQN(OBS_SIZE, N_ACTIONS).to(DEVICE)
        self._load_model(model_path)
        self.policy.eval()

        # Environment
        self.env = WarehouseMultiEnv(render_mode="human" if render else None)

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        ckpt  = torch.load(path, map_location=DEVICE)
        state = ckpt.get("policy", ckpt)
        self.policy.load_state_dict(state)

        # Print checkpoint metadata if available
        if isinstance(ckpt, dict) and "episode" in ckpt:
            print(f"  Checkpoint from episode : {ckpt['episode']}")
            print(f"  Best score during train : {ckpt.get('best_score', 'N/A')}")
            print(f"  Epsilon at save         : {ckpt.get('epsilon', 'N/A'):.4f}")
        print(f"  Model loaded from       : {path}\n")

    def _select_actions(self, obs_dict: dict) -> dict:
        """Pure greedy — no random actions during evaluation."""
        greedy_obs = [obs_dict[a] for a in self.agents]
        obs_batch  = torch.from_numpy(np.stack(greedy_obs)).to(DEVICE)
        with torch.no_grad():
            q_vals = self.policy(obs_batch)
        best = q_vals.argmax(dim=1).cpu().tolist()
        return {agent: int(act) for agent, act in zip(self.agents, best)}

    def run(self, n_episodes: int = 10) -> dict:
        """
        Run n_episodes fully greedy evaluation episodes.
        Returns a summary dict with all collected metrics.
        """
        print("=" * 60)
        print(f"  EVALUATION — {n_episodes} episodes, {NUM_AGENTS} robots")
        print(f"  Goal per episode: {10} deliveries")
        print("=" * 60)
        print(f"{'Ep':>4}  {'Score':>6}  {'Steps':>6}  {'AvgR':>8}  "
              f"{'DelivRate':>10}  {'Time(s)':>8}")
        print("-" * 60)

        # Metrics collectors
        scores        = []
        steps_list    = []
        avg_rewards   = []
        delivery_rates = []
        times         = []
        per_robot_rewards = {a: [] for a in self.agents}
        action_counts = {a: [0]*N_ACTIONS for a in self.agents}
        success_eps   = 0

        for ep in range(n_episodes):
            obs, _     = self.env.reset()
            ep_rewards = {a: 0.0 for a in self.agents}
            ep_steps   = 0
            t_start    = time.time()

            while self.env.agents:
                actions  = self._select_actions(obs)

                # Track action distribution
                for agent in self.agents:
                    action_counts[agent][actions[agent]] += 1

                obs, rewards, terms, truncs, _ = self.env.step(actions)

                for agent in self.agents:
                    ep_rewards[agent] += rewards[agent]

                ep_steps += 1
                if ep_steps > 16000:
                    break

            elapsed        = time.time() - t_start
            score          = self.env.score
            goal           = self.env.goal_deliveries
            delivery_rate  = score / goal * 100
            avg_r          = sum(ep_rewards.values()) / NUM_AGENTS

            scores.append(score)
            steps_list.append(ep_steps)
            avg_rewards.append(avg_r)
            delivery_rates.append(delivery_rate)
            times.append(elapsed)

            for agent in self.agents:
                per_robot_rewards[agent].append(ep_rewards[agent])

            if score >= goal:
                success_eps += 1

            print(f"{ep+1:>4}  {score:>4}/{goal:<2}  {ep_steps:>6}  "
                  f"{avg_r:>8.1f}  {delivery_rate:>9.1f}%  {elapsed:>7.1f}s")

        self.env.close()

        # ── Summary Report ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  SUMMARY REPORT")
        print("=" * 60)

        print(f"\n  Episodes run        : {n_episodes}")
        print(f"  Successful episodes : {success_eps} / {n_episodes}  "
              f"({success_eps/n_episodes*100:.1f}%)")

        print(f"\n  Score")
        print(f"    Mean   : {np.mean(scores):.2f} / {self.env.goal_deliveries}")
        print(f"    Max    : {np.max(scores)}")
        print(f"    Min    : {np.min(scores)}")
        print(f"    Std    : {np.std(scores):.2f}")

        print(f"\n  Delivery rate")
        print(f"    Mean   : {np.mean(delivery_rates):.1f}%")
        print(f"    Best   : {np.max(delivery_rates):.1f}%")
        print(f"    Worst  : {np.min(delivery_rates):.1f}%")

        print(f"\n  Steps per episode")
        print(f"    Mean   : {np.mean(steps_list):.0f}")
        print(f"    Min    : {np.min(steps_list)}  (faster = better)")
        print(f"    Max    : {np.max(steps_list)}")

        print(f"\n  Avg reward per robot per episode")
        print(f"    Mean   : {np.mean(avg_rewards):.1f}")
        print(f"    Best   : {np.max(avg_rewards):.1f}")

        print(f"\n  Wall time")
        print(f"    Total  : {sum(times):.1f}s")
        print(f"    Per ep : {np.mean(times):.1f}s")

        # Per-robot breakdown
        print(f"\n  Per-robot avg reward")
        for agent in self.agents:
            r = np.mean(per_robot_rewards[agent])
            print(f"    {agent:>8} : {r:.1f}")

        # Action distribution
        action_names = ["Up", "Down", "Left", "Right", "Interact", "Wait"]
        print(f"\n  Action distribution (across all robots, all episodes)")
        total_actions = sum(
            sum(action_counts[a]) for a in self.agents
        )
        combined = [0] * N_ACTIONS
        for a in self.agents:
            for i in range(N_ACTIONS):
                combined[i] += action_counts[a][i]
        for i, name in enumerate(action_names):
            pct = combined[i] / total_actions * 100
            bar = "█" * int(pct / 2)
            print(f"    {name:>9} : {pct:5.1f}%  {bar}")

        print("\n" + "=" * 60)

        return {
            "scores":          scores,
            "steps":           steps_list,
            "delivery_rates":  delivery_rates,
            "avg_rewards":     avg_rewards,
            "success_rate":    success_eps / n_episodes,
            "action_counts":   action_counts,
        }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained IDQN agent")
    parser.add_argument("--model",    type=str, default="checkpoints/best_policy.pt",
                        help="Path to .pt model or checkpoint file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes to run")
    parser.add_argument("--render",   action="store_true",
                        help="Render the environment visually (slower)")
    parser.add_argument("--no-render", dest="render", action="store_false")

    # Change render on/ off 
    parser.set_defaults(render=True)
    args = parser.parse_args()

    evaluator = AgentEvaluator(model_path=args.model, render=args.render)
    evaluator.run(n_episodes=args.episodes)