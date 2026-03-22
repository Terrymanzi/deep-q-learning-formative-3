# play.py — Evaluate a trained DQN model on ALE/Breakout-v5
# Loads a saved model, renders gameplay live, and reports per-episode rewards.

import os
import argparse

import ale_py  # noqa: F401  (registers ALE envs with gymnasium)
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


# ─── DEFAULTS ────────────────────────────────────────────────────────────────

ENV_ID      = "ALE/Breakout-v5"
FRAME_STACK = 4
N_EPISODES  = 5          # number of full episodes to play
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),   # src/
    "..", "results", "dqn_model" # → results/dqn_model.zip
)


# ─── ENVIRONMENT ─────────────────────────────────────────────────────────────

def make_play_env(render: bool = True) -> VecTransposeImage:
    """
    Build the same VecEnv stack used during training, optionally with
    render_mode='human' so the game window opens automatically.
    """
    env_kwargs = {"render_mode": "human"} if render else {}
    env = make_atari_env(ENV_ID, n_envs=1, seed=0, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)
    return env


# ─── EVALUATION LOOP ─────────────────────────────────────────────────────────

def play(model_path: str, n_episodes: int = N_EPISODES, render: bool = True) -> None:
    # ── load model ──
    model_path = os.path.splitext(model_path)[0]   # strip .zip if present
    print(f"\n  Loading model : {model_path}.zip")
    model = DQN.load(model_path)
    print(f"  Policy        : {model.policy.__class__.__name__}")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Deterministic : True  (greedy policy)\n")

    env = make_play_env(render=render)

    episode_rewards = []
    current_ep_reward = 0.0
    episodes_done = 0

    obs = env.reset()   # VecEnv reset → (1, C, H, W) array

    print(f"  {'Episode':>8}  {'Reward':>10}")
    print("  " + "─" * 22)

    while episodes_done < n_episodes:
        # Greedy action from the trained Q-network
        action, _ = model.predict(obs, deterministic=True)

        # VecEnv step — renders automatically when render_mode='human'
        obs, rewards, dones, infos = env.step(action)
        current_ep_reward += float(rewards[0])

        if dones[0]:
            episodes_done += 1
            episode_rewards.append(current_ep_reward)
            print(f"  {episodes_done:>8}  {current_ep_reward:>10.2f}")
            current_ep_reward = 0.0
            # VecEnv auto-resets; obs is already the first frame of the new episode

    env.close()

    # ── summary ──
    print("  " + "─" * 22)
    print(f"\n  Mean reward over {n_episodes} episodes : {np.mean(episode_rewards):.2f}")
    print(f"  Std                                 : {np.std(episode_rewards):.2f}")
    print(f"  Min / Max                           : {min(episode_rewards):.2f} / {max(episode_rewards):.2f}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play a trained DQN on Breakout.")
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to .zip model file (default: results/dqn_model.zip)",
    )
    p.add_argument(
        "--episodes", type=int, default=N_EPISODES,
        help=f"Number of episodes to run (default: {N_EPISODES})",
    )
    p.add_argument(
        "--no-render", action="store_true",
        help="Disable the game window (useful for headless evaluation)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    play(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render,
    )
