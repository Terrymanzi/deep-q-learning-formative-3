# play.py — Load and Evaluate a trained DQN model on ALE/Breakout-v5 with version compatibility fallbacks
# Loads a saved model, renders gameplay live, and reports per-episode rewards.
import os
import argparse
import zipfile
import numpy as np
import torch
import pickle

import ale_py
from gymnasium import Env
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


ENV_ID      = "ALE/Breakout-v5"
FRAME_STACK = 4
N_EPISODES  = 5
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),
    "..", "results", "dqn_model"
)


def make_play_env(render: bool = True) -> VecTransposeImage:
    """Build the same VecEnv stack used during training."""
    env_kwargs = {"render_mode": "human"} if render else {}
    env = make_atari_env(ENV_ID, n_envs=1, seed=0, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)
    return env


def load_model_compatible(model_path: str):
    """Load model with compatibility workaround for version mismatches."""
    model_path = os.path.splitext(model_path)[0]
    zip_path = model_path + ".zip"

    print(f"Loading model from: {zip_path}")

    # Try to load with standard method first
    try:
        model = DQN.load(model_path, device="cpu")
        return model
    except (ValueError, KeyError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"Standard load failed with: {type(e).__name__}")

    # Create environment and load model with minimal setup
    print("Attempting with minimal setup...")
    env_temp = make_play_env(render=False)

    # Create model with minimal buffer size to avoid memory issues
    model_temp = DQN(
        "CnnPolicy",
        env_temp,
        device="cpu",
        verbose=0,
        buffer_size=100,  # Minimal buffer
        learning_starts=0,
    )
    env_temp.close()

    # Now load just the policy weights from the zip
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Load policy state dict only (skip optimizer)
            with zf.open('policy.pth') as f:
                policy_state = torch.load(f, map_location="cpu")
                model_temp.policy.load_state_dict(policy_state)
                print("Successfully loaded policy network")
                return model_temp
    except Exception as e:
        print(f"Manual policy loading failed: {e}")
        raise ValueError(f"Could not load model with any method")


def play(model_path: str, n_episodes: int = N_EPISODES, render: bool = True) -> None:
    print(f"\n  Loading model : {os.path.splitext(model_path)[0]}.zip")
    model = load_model_compatible(model_path)
    print(f"  Policy        : {model.policy.__class__.__name__}")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Deterministic : True  (greedy policy)\n")

    env = make_play_env(render=render)

    episode_rewards = []
    current_ep_reward = 0.0
    episodes_done = 0

    obs = env.reset()

    print(f"  {'Episode':>8}  {'Reward':>10}")
    print("  " + "-" * 22)

    while episodes_done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        current_ep_reward += float(rewards[0])

        if dones[0]:
            episodes_done += 1
            episode_rewards.append(current_ep_reward)
            print(f"  {episodes_done:>8}  {current_ep_reward:>10.2f}")
            current_ep_reward = 0.0

    env.close()

    print("  " + "-" * 22)
    print(f"\n  Mean reward over {n_episodes} episodes : {np.mean(episode_rewards):.2f}")
    print(f"  Std                                 : {np.std(episode_rewards):.2f}")
    print(f"  Min / Max                           : {min(episode_rewards):.2f} / {max(episode_rewards):.2f}\n")


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
