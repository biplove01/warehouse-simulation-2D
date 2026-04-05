"""
trainer_idqn.py — IDQN with Parameter Sharing for WarehouseMultiEnv

All robots share ONE DQN network and ONE replay buffer.
This means:
  - 1x memory regardless of robot count
  - Every robot's experience improves the shared policy
  - Adding more robots = more data, not more networks

Usage:
  python trainer_idqn.py --mode train --episodes 5000
  python trainer_idqn.py --mode human --model checkpoints/best_policy.pt
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from warehouse_multi_env import WarehouseMultiEnv, NUM_AGENTS, OBS_SIZE, N_ACTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Network ───────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    """
    Dueling DQN architecture.
    Separates value V(s) from advantage A(s,a) — trains faster on sparse rewards.
    LayerNorm after each hidden layer stabilises training with mixed-scale rewards.
    """

    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),      nn.LayerNorm(256), nn.ReLU(),
        )
        # Value stream: scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream: A(s,a) per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        V    = self.value_stream(feat)          # (B, 1)
        A    = self.advantage_stream(feat)      # (B, n_actions)
        # Q = V + (A - mean(A))  — removes identifiability issue
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Shared experience replay — all robots push to the same buffer."""

    def __init__(self, capacity: int = 150_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((
            np.array(obs,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.from_numpy(np.stack(obs)).to(DEVICE),
            torch.tensor(act,  dtype=torch.long).unsqueeze(1).to(DEVICE),
            torch.tensor(rew,  dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.from_numpy(np.stack(nobs)).to(DEVICE),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ── Trainer ───────────────────────────────────────────────────────────────────

class IDQNTrainer:
    """
    IDQN with:
      - Parameter sharing across all agents
      - Double DQN (policy selects action, target evaluates)
      - Dueling architecture (see DQN above)
      - Linear epsilon decay
      - Periodic target network sync
      - Checkpoint saving
    """

    def __init__(
        self,
        env: WarehouseMultiEnv,
        lr:               float = 3e-4,
        gamma:            float = 0.99,
        batch_size:       int   = 256,
        buffer_capacity:  int   = 150_000,
        warmup_steps:     int   = 2_000,    # steps before training starts
        target_sync_freq: int   = 400,      # gradient steps between target syncs
        epsilon_start:    float = 1.0,
        epsilon_end:      float = 0.05,
        epsilon_decay:    float = 0.9997,   # per episode
    ):
        self.env        = env
        self.gamma      = gamma
        self.batch_size = batch_size
        self.warmup     = warmup_steps
        self.sync_freq  = target_sync_freq
        self.eps        = epsilon_start
        self.eps_end    = epsilon_end
        self.eps_decay  = epsilon_decay
        self.agents     = [f"robot_{i}" for i in range(NUM_AGENTS)]

        # ONE shared policy + target
        self.policy = DQN(OBS_SIZE, N_ACTIONS).to(DEVICE)
        self.target = DQN(OBS_SIZE, N_ACTIONS).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt    = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self._grad_steps  = 0
        self._total_steps = 0

    # ── action selection ──────────────────────────────────────────────────────

    def select_actions(self, obs_dict: dict) -> dict:
        actions = {}
        for agent in self.agents:
            if random.random() < self.eps:
                actions[agent] = self.env.action_space(agent).sample()
            else:
                obs_t = torch.from_numpy(obs_dict[agent]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    q = self.policy(obs_t)
                actions[agent] = int(q.argmax(dim=1).item())
        return actions

    # ── gradient step ─────────────────────────────────────────────────────────

    def _train_step(self) -> float | None:
        if len(self.buffer) < self.warmup:
            return None

        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)

        # Current Q for chosen actions
        current_q = self.policy(obs).gather(1, act)

        # Double DQN target
        with torch.no_grad():
            next_act   = self.policy(nobs).argmax(dim=1, keepdim=True)
            next_q     = self.target(nobs).gather(1, next_act)
            target_q   = rew + self.gamma * next_q * (1.0 - done)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        self._grad_steps += 1
        if self._grad_steps % self.sync_freq == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return loss.item()

    # ── training loop ─────────────────────────────────────────────────────────

    def run(
        self,
        n_episodes: int  = 5000,
        save_dir:   str  = "checkpoints",
        save_every: int  = 50,
        log_every:  int  = 2,
    ):
        os.makedirs(save_dir, exist_ok=True)
        best_score   = -1
        score_window = deque(maxlen=100)   # for rolling average

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            ep_rewards = {a: 0.0 for a in self.agents}
            ep_losses  = []
            ep_steps   = 0

            while self.env.agents:
                actions = self.select_actions(obs)
                next_obs, rewards, terms, truncs, _ = self.env.step(actions)

                for agent in self.agents:
                    done_flag = float(terms[agent] or truncs[agent])
                    self.buffer.push(
                        obs[agent], actions[agent],
                        rewards[agent], next_obs[agent], done_flag
                    )
                    ep_rewards[agent] += rewards[agent]

                loss = self._train_step()
                if loss is not None:
                    ep_losses.append(loss)

                obs               = next_obs
                self._total_steps += 1
                ep_steps          += 1

                if ep_steps > 16000:   # hard safety cap
                    break

            # Epsilon decay (per episode)
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

            score = self.env.score
            score_window.append(score)
            avg_score = sum(score_window) / len(score_window)
            avg_r     = sum(ep_rewards.values()) / NUM_AGENTS
            avg_loss  = sum(ep_losses) / max(len(ep_losses), 1)

            # ── Logging ──
            if ep % log_every == 0:
                print(
                    f"Ep {ep:5d} | score={score:3d}/{self.env.goal_deliveries}"
                    f" | avg100={avg_score:5.2f}"
                    f" | avg_r={avg_r:8.1f}"
                    f" | loss={avg_loss:.4f}"
                    f" | ε={self.eps:.3f}"
                    f" | steps={ep_steps}"
                    f" | buf={len(self.buffer):,}"
                )

            # ── Checkpointing ──
            if score > best_score:
                best_score = score
                torch.save(self.policy.state_dict(),
                           os.path.join(save_dir, "best_policy.pt"))

            if ep % save_every == 0 and ep > 0:
                ckpt = {
                    "policy":    self.policy.state_dict(),
                    "target":    self.target.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "epsilon":   self.eps,
                    "episode":   ep,
                    "best_score": best_score,
                }
                torch.save(ckpt, os.path.join(save_dir, f"ckpt_ep{ep:05d}.pt"))
                print(f"  [saved checkpoint at ep {ep}, best={best_score}]")

        print(f"\nTraining complete. Best score: {best_score}  "
              f"Total env steps: {self._total_steps:,}")
        return self.policy

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        # Support both raw state-dict and full checkpoint dict
        state = ckpt.get("policy", ckpt)
        self.policy.load_state_dict(state)
        self.target.load_state_dict(state)
        if "epsilon" in ckpt:
            self.eps = ckpt["epsilon"]
        print(f"Loaded policy from {path}  (ε={self.eps:.3f})")


# ── Human playback ────────────────────────────────────────────────────────────

def run_human(model_path: str | None = None):
    """Watch the trained policy drive all robots."""
    env     = WarehouseMultiEnv(render_mode="human")
    trainer = IDQNTrainer(env)
    if model_path:
        trainer.load(model_path)
    trainer.eps = 0.0   # pure exploitation — no random actions

    obs, _ = env.reset()
    try:
        while True:
            actions = trainer.select_actions(obs)
            obs, _, terms, truncs, _ = env.step(actions)
            if not env.agents:
                print(f"Episode done — score: {env.score}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# ── Curriculum helper ─────────────────────────────────────────────────────────

def run_curriculum(save_dir: str = "checkpoints"):
    """
    Two-phase curriculum:
      Phase 1 — 2000 episodes, 2 robots, 5 packages → learn basic navigation
      Phase 2 — 5000 episodes, 5 robots, 10 packages → full task
    Transfer: policy weights from phase 1 initialise phase 2.
    """
    import warehouse_multi_env as em

    # ── Phase 1 ──
    print("=" * 60)
    print("PHASE 1: 2 robots, 5 packages")
    print("=" * 60)

    orig_num   = em.NUM_AGENTS
    orig_obs   = em.OBS_SIZE
    em.NUM_AGENTS = 2
    em.OBS_SIZE   = 7 + (2 - 1) * 3 + 8   # 18

    env1    = WarehouseMultiEnv(render_mode=None)
    # Monkey-patch: 5 packages only
    orig_reset = env1.reset
    def patched_reset(**kw):
        obs, info = orig_reset(**kw)
        # trim queue to 5
        while len(env1.target_queue) > 5:
            env1.target_queue.pop()
        env1.goal_deliveries = 5
        return obs, info
    env1.reset = patched_reset

    t1 = IDQNTrainer(env1, epsilon_decay=0.9995)
    t1.run(n_episodes=2000, save_dir=save_dir, log_every=100)

    # ── Phase 2 ──
    print("\n" + "=" * 60)
    print("PHASE 2: 5 robots, 10 packages (full task)")
    print("=" * 60)

    em.NUM_AGENTS = orig_num
    em.OBS_SIZE   = orig_obs

    env2 = WarehouseMultiEnv(render_mode=None)
    t2   = IDQNTrainer(env2, epsilon_start=0.3)   # lower epsilon: reuse phase-1 knowledge

    # Transfer shared layers from phase-1 (input layer size differs, so partial)
    p1_state = t1.policy.state_dict()
    p2_state = t2.policy.state_dict()
    for key in p2_state:
        if key in p1_state and p1_state[key].shape == p2_state[key].shape:
            p2_state[key] = p1_state[key]
    t2.policy.load_state_dict(p2_state)
    t2.target.load_state_dict(p2_state)

    t2.run(n_episodes=5000, save_dir=save_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDQN Warehouse Trainer")
    parser.add_argument("--mode",       choices=["train","human","curriculum"],
                        default="train")
    parser.add_argument("--model",      type=str, default=None,
                        help="Path to .pt checkpoint to load")
    parser.add_argument("--episodes",   type=int, default=5000)
    parser.add_argument("--save-dir",   type=str, default="checkpoints")
    parser.add_argument("--log-every",  type=int, default=50)
    args = parser.parse_args()

    if args.mode == "train":
        env     = WarehouseMultiEnv(render_mode=None)
        trainer = IDQNTrainer(env)
        if args.model:
            trainer.load(args.model)
        trainer.run(n_episodes=args.episodes,
                    save_dir=args.save_dir,
                    log_every=args.log_every)

    elif args.mode == "human":
        run_human(model_path=args.model)

    elif args.mode == "curriculum":
        run_curriculum(save_dir=args.save_dir)