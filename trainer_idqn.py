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
import pickle



import numpy as np
import torch
import torch.nn as nn
from collections import deque

from warehouse_multi_env import WarehouseMultiEnv, NUM_AGENTS, OBS_SIZE, N_ACTIONS

# ── GPU optimisations ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True          # auto-tune kernels for your GPU
    torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on Ampere+ GPUs
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


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
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q



# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Shared experience replay — all robots push to the same buffer.
    Uses pinned memory for faster CPU→GPU transfers.
    """

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

        # pin_memory() + non_blocking=True → async CPU→GPU transfer
        obs_t  = torch.from_numpy(np.stack(obs)).pin_memory().to(DEVICE, non_blocking=True)
        act_t  = torch.tensor(act,  dtype=torch.long).unsqueeze(1).pin_memory().to(DEVICE, non_blocking=True)
        rew_t  = torch.tensor(rew,  dtype=torch.float32).unsqueeze(1).pin_memory().to(DEVICE, non_blocking=True)
        nobs_t = torch.from_numpy(np.stack(nobs)).pin_memory().to(DEVICE, non_blocking=True)
        done_t = torch.tensor(done, dtype=torch.float32).unsqueeze(1).pin_memory().to(DEVICE, non_blocking=True)

        return obs_t, act_t, rew_t, nobs_t, done_t

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
      - GPU optimisations: pinned memory, batched action selection,
        multiple gradient updates per env step
    """

    def __init__(
        self,
        env: WarehouseMultiEnv,
        lr:               float = 3e-4,
        gamma:            float = 0.99,
        batch_size:       int   = 1024,    # ↑ from 256 — keeps GPU fed
        buffer_capacity:  int   = 150_000,
        warmup_steps:     int   = 2_000,
        target_sync_freq: int   = 400,
        epsilon_start:    float = 1.0,
        epsilon_end:      float = 0.05,
        epsilon_decay:    float = 0.9997,
        grad_updates_per_step: int = 4,    # multiple gradient steps per env step
    ):
        self.env        = env
        self.gamma      = gamma
        self.batch_size = batch_size
        self.warmup     = warmup_steps
        self.sync_freq  = target_sync_freq
        self.eps        = epsilon_start
        self.eps_end    = epsilon_end
        self.eps_decay  = epsilon_decay
        self.grad_updates_per_step = grad_updates_per_step
        self.agents     = [f"robot_{i}" for i in range(NUM_AGENTS)]

        # ONE shared policy + target
        # Note: torch.compile is disabled — Triton is not supported on Windows
        self.policy = DQN(OBS_SIZE, N_ACTIONS).to(DEVICE)
        self.target = DQN(OBS_SIZE, N_ACTIONS).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt    = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self._grad_steps  = 0
        self._total_steps = 0

    # ── action selection (batched — one GPU call for all agents) ─────────────

    def save_buffer(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer.buf), f)
        print(f"Buffer saved ({len(self.buffer)} transitions)")

    def load_buffer(self, path: str):
        if not os.path.exists(path):
            print("No buffer file found, starting fresh.")
            return
        with open(path, "rb") as f:
            transitions = pickle.load(f)
        for t in transitions:
            self.buffer.buf.append(t)
        print(f"Buffer loaded ({len(self.buffer)} transitions)")

    def select_actions(self, obs_dict: dict) -> dict:
        """
        Stacks all observations into a single batch and runs ONE forward pass
        for all robots at once instead of one GPU call per robot.
        """
        actions = {}
        greedy_agents = []
        greedy_obs    = []

        for agent in self.agents:
            if random.random() < self.eps:
                actions[agent] = self.env.action_space(agent).sample()
            else:
                greedy_agents.append(agent)
                greedy_obs.append(obs_dict[agent])

        if greedy_agents:
            obs_batch = torch.from_numpy(np.stack(greedy_obs)).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                q_vals = self.policy(obs_batch)
            best_actions = q_vals.argmax(dim=1).cpu().tolist()
            for agent, action in zip(greedy_agents, best_actions):
                actions[agent] = int(action)

        return actions


    # ── gradient step ─────────────────────────────────────────────────────────

    def _train_step(self) -> float | None:
        if len(self.buffer) < self.warmup:
            return None

        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)

        current_q = self.policy(obs).gather(1, act)

        with torch.no_grad():
            next_act = self.policy(nobs).argmax(dim=1, keepdim=True)
            next_q   = self.target(nobs).gather(1, next_act)
            target_q = rew + self.gamma * next_q * (1.0 - done)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.opt.zero_grad(set_to_none=True)   # faster than zero_grad()
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
            n_episodes: int = 5000,
            save_dir: str = "checkpoints",
            save_every: int = 10,
            log_every: int = 2,
            agent_stagnation_limit: int = 600,
            max_ep_steps: int = 3000,
            heartbeat_every: int = 200,  # ← print a mid-episode pulse every N steps
    ):
        import time
        os.makedirs(save_dir, exist_ok=True)
        best_score = -1

        score_window = deque(maxlen=100)
        loss_window = deque(maxlen=100)

        total_success, total_fail, total_timeout = 0, 0, 0

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            ep_losses = []
            ep_steps = 0
            max_current_stagnation = 0
            ep_start = time.time()

            deliveries_per_agent = {a: 0 for a in self.agents}
            stagnant_steps_per_agent = {a: 0 for a in self.agents}
            active_agents = set(self.agents)

            while active_agents and ep_steps < max_ep_steps:

                # ── Heartbeat: print a pulse every heartbeat_every step ──────────
                if ep_steps > 0 and ep_steps % heartbeat_every == 0:
                    elapsed = time.time() - ep_start
                    avg_loss = sum(loss_window) / len(loss_window) if loss_window else 0.0
                    buf_pct = min(100, 100 * len(self.buffer) / self.warmup)
                    print(
                        f"  [Ep {ep} | step {ep_steps}/{max_ep_steps}] "
                        f"active={len(active_agents)} "
                        f"buf={len(self.buffer)}({buf_pct:.0f}%warm) "
                        f"loss={avg_loss:.5f} eps={self.eps:.3f} "
                        f"stag={max_current_stagnation} "
                        f"elapsed={elapsed:.1f}s"
                    )

                # ── Action selection (only active agents) ─────────────────────────
                current_obs = {a: obs[a] for a in active_agents if a in obs}
                actions = self.select_actions(current_obs)

                # ── Env step ──────────────────────────────────────────────────────
                next_obs, rewards, terms, truncs, _ = self.env.step(actions)

                # ── Experience processing ─────────────────────────────────────────
                for agent in list(active_agents):
                    if agent not in rewards:
                        active_agents.discard(agent)
                        continue

                    stagnant_steps_per_agent[agent] += 1
                    if rewards[agent] > 0.5:
                        deliveries_per_agent[agent] += 1
                        stagnant_steps_per_agent[agent] = 0

                    done_flag = float(terms[agent] or truncs[agent])
                    self.buffer.push(
                        obs[agent], actions[agent],
                        rewards[agent], next_obs[agent], done_flag
                    )

                    if terms[agent] or truncs[agent]:
                        active_agents.discard(agent)

                # ── Stagnation exit ───────────────────────────────────────────────
                max_current_stagnation = max(stagnant_steps_per_agent.values())
                if max_current_stagnation >= agent_stagnation_limit:
                    print(
                        f"  [Ep {ep} | step {ep_steps}] "
                        f"Stagnation break — worst robot stuck for "
                        f"{max_current_stagnation} steps"
                    )
                    break

                # ── Gradient updates (skip entirely during warmup) ────────────────
                if len(self.buffer) >= self.warmup:
                    for _ in range(self.grad_updates_per_step):
                        loss = self._train_step()
                        if loss is not None:
                            ep_losses.append(loss)
                            loss_window.append(loss)

                obs = next_obs
                ep_steps += 1
                self._total_steps += 1

            # ── Post-episode stats ────────────────────────────────────────────────
            final_score = self.env.score
            ep_duration = time.time() - ep_start
            score_window.append(final_score)

            if final_score >= self.env.goal_deliveries:
                total_success += 1
            elif ep_steps >= max_ep_steps:
                total_timeout += 1
            else:
                total_fail += 1

            self.eps = max(self.eps_end, self.eps * self.eps_decay)

            # ── Episode log ───────────────────────────────────────────────────────
            if ep % log_every == 0:
                avg_loss = sum(loss_window) / len(loss_window) if loss_window else 0.0
                delivery_info = " | ".join(
                    [f"R{i}:{deliveries_per_agent[a]}" for i, a in enumerate(self.agents)]
                )

                print(f"\n{'=' * 40}")
                print(f"Episode {ep:>5}  ({ep_duration:.1f}s)")
                print(f"  Score      : {final_score}/{self.env.goal_deliveries}  "
                      f"(Avg100: {np.mean(score_window):.2f})")
                print(f"  Loss       : {avg_loss:.5f}")
                print(f"  Epsilon    : {self.eps:.4f}")
                print(f"  Steps      : {ep_steps}  |  Grad steps: {self._grad_steps}")
                print(f"  Stagnation : {max_current_stagnation}")
                print(f"  Deliveries : {delivery_info}")
                print(f"  Buffer     : {len(self.buffer)}/{self.buffer.buf.maxlen}")
                print(f"  Outcomes   : ✓{total_success}  ✗stag:{total_fail}  ⌛{total_timeout}")
                if DEVICE.type == "cuda":
                    print(f"  VRAM       : {torch.cuda.memory_allocated() / 1e6:.0f} MB")
                print(f"{'=' * 40}\n")

            # ── Checkpointing ─────────────────────────────────────────────────────
            print(f"  ★ Final score: {final_score}.")

            if final_score > best_score:
                best_score = final_score
                torch.save(self.policy.state_dict(), os.path.join(save_dir, "best_policy.pt"))
                print(f"  ★ New best score: {best_score} — checkpoint saved")
                self.save_buffer(os.path.join(save_dir, "best_policy_buffer.pkl"))  # ← add this

            if ep % save_every == 0 and ep > 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_ep{ep:05d}.pt")
                # torch.save({"policy": self.policy.state_dict(),
                #             "epsilon": self.eps, "episode": ep}, ckpt_path)
                torch.save({
                    "policy": self.policy.state_dict(),
                    "epsilon": self.eps,
                    "episode": ep
                }, ckpt_path)
                self.save_buffer(ckpt_path.replace(".pt", "_buffer.pkl"))

        return self.policy

    # ── load ──────────────────────────────────────────────────────────────────

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        state = ckpt.get("policy", ckpt)
        self.policy.load_state_dict(state)
        self.target.load_state_dict(state)
        if "epsilon" in ckpt:
            self.eps = ckpt["epsilon"]
        self.load_buffer(path.replace(".pt", "_buffer.pkl"))
        print(f"Loaded policy from {path}  (ε={self.eps:.3f})")

# ── Human playback ────────────────────────────────────────────────────────────

def run_human(model_path: str | None = None):
    """Watch the trained policy drive all robots."""
    env     = WarehouseMultiEnv(render_mode="human")
    trainer = IDQNTrainer(env)
    if model_path:
        trainer.load(model_path)
    trainer.eps = 0.0

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
    """
    import warehouse_multi_env as em

    print("=" * 60)
    print("PHASE 1: 2 robots, 5 packages")
    print("=" * 60)

    orig_num   = em.NUM_AGENTS
    orig_obs   = em.OBS_SIZE
    em.NUM_AGENTS = 2
    em.OBS_SIZE   = 7 + (2 - 1) * 3 + 8

    env1       = WarehouseMultiEnv(render_mode=None)
    orig_reset = env1.reset
    def patched_reset(**kw):
        obs, info = orig_reset(**kw)
        while len(env1.target_queue) > 5:
            env1.target_queue.pop()
        env1.goal_deliveries = 5
        return obs, info
    env1.reset = patched_reset

    t1 = IDQNTrainer(env1, epsilon_decay=0.9995)
    t1.run(n_episodes=2000, save_dir=save_dir, log_every=10)

    print("\n" + "=" * 60)
    print("PHASE 2: 5 robots, 10 packages (full task)")
    print("=" * 60)

    em.NUM_AGENTS = orig_num
    em.OBS_SIZE   = orig_obs

    env2 = WarehouseMultiEnv(render_mode=None)
    t2   = IDQNTrainer(env2, epsilon_start=0.3)

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
    parser.add_argument("--log-every",  type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    if args.mode == "train":
        env     = WarehouseMultiEnv(render_mode=None)
        trainer = IDQNTrainer(env)

        model_path = args.model or os.path.join(args.save_dir, "best_policy.pt")
        if os.path.exists(model_path):
            print(f"Resuming from {model_path}")
            trainer.load(model_path)
        else:
            print("No checkpoint found — starting fresh")

        trainer.run(
            n_episodes=args.episodes,
            save_dir=args.save_dir,
            log_every=args.log_every,
            save_every=args.save_every,
        )

    elif args.mode == "human":
        run_human(model_path=args.model)

    elif args.mode == "curriculum":
        run_curriculum(save_dir=args.save_dir)

