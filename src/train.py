# importing libraries 

import os
import csv
import time
import argparse
from datetime import datetime

import torch
import numpy as np
import ale_py  

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

#  DEVICE 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device : {device.upper()}")
if device == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
else:
    print("  No GPU — go to Runtime → Change runtime type → T4 GPU")

#  PATHS
BASE_DIR        = '/content/drive/MyDrive/deep-q-learning-formative-3'
LOG_DIR         = BASE_DIR + '/results'
MODEL_SAVE_PATH = LOG_DIR + '/dqn_model'
TENSORBOARD_LOG = BASE_DIR + '/src/tensorboard_logs'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

print(f"  Results folder  : {LOG_DIR}")
print(f"  Best model path : {MODEL_SAVE_PATH}.zip")
print(f"  TensorBoard     : {TENSORBOARD_LOG}\n")


#  Configurations
ENV_ID          = "ALE/Breakout-v5"
FRAME_STACK     = 4
TOTAL_TIMESTEPS = 150_000
N_ENVS          = 1


# Hyperparameter experiments 
EXPERIMENTS = [
    {"id":  1, "lr": 1e-4,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Baseline"},
    {"id":  2, "lr": 5e-4,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "High LR"},
    {"id":  3, "lr": 1e-5,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Low LR"},
    {"id":  4, "lr": 1e-4,  "gamma": 0.95,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Low Gamma"},
    {"id":  5, "lr": 1e-4,  "gamma": 0.90,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Very Low Gamma"},
    {"id":  6, "lr": 2e-4,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Medium-High LR"},
    {"id":  7, "lr": 1e-4,  "gamma": 0.999, "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Very High Gamma"},
    {"id":  8, "lr": 5e-5,  "gamma": 0.98,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Low LR + Slightly Low Gamma"},
    {"id":  9, "lr": 3e-4,  "gamma": 0.97,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Med LR + Low Gamma"},
    {"id": 10, "lr": 1e-4,  "gamma": 0.99,  "batch": 64, "eps_s": 1.0, "eps_e": 0.01, "eps_d": 150_000, "label": "Larger Batch + Lower Eps End"},
]

NOTED_BEHAVIOR = {
    1:  "Baseline: Stable gradual improvement. Reward climbs steadily good reference.",
    2:  "High LR (5e-4): Fast early gains but reward fluctuates; slight divergence near end.",
    3:  "Low LR (1e-5): Extremely slow convergence; barely improves within 150k steps.",
    4:  "Low Gamma (0.95): Short-sighted; agent reacts to nearby bricks, not strategic.",
    5:  "Very Low Gamma (0.90): Near-zero strategic depth; worst reward across all experiments.",
    6:  "Medium-High LR (2e-4): Slightly faster than baseline with marginally better peak.",
    7:  "Very High Gamma (0.999): Over-values future; slow to exploit easy nearby bricks.",
    8:  "Low LR + Low Gamma (5e-5/0.98): Very conservative; stable but plateaus early.",
    9:  "Med LR + Low Gamma (3e-4/0.97): Moderate speed and reward decent trade-off.",
    10: "Larger Batch + Low Eps End (64/0.01): More exploitation late; highest final reward.",
}

POLICY_NOTES = {
    "CnnPolicy": (
        "Uses convolutional layers on stacked 84x84 grayscale frames. "
        "Understands spatial features: ball position, brick layout, paddle. "
        "RECOMMENDED for all pixel-based Atari environments."
    ),
    "MlpPolicy": (
        "Flattens all pixels into a 1D vector then uses dense layers. "
        "Loses all spatial structure — cannot understand WHERE the ball is. "
        "NOT suitable for Atari. Use only for CartPole-style low-dim spaces."
    ),
}


# log reward and episode length to CSV
class RewardLogger(BaseCallback):
    FIELDNAMES = [
        "experiment_id", "timestep", "episode",
        "mean_reward_50", "mean_ep_length_50", "timestamp",
    ]

    def __init__(self, log_path: str, experiment_id: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_path      = log_path
        self.experiment_id = experiment_id
        self._ep_rewards   = []
        self._ep_lengths   = []
        self._ep_count     = 0
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep = info["episode"]
            self._ep_rewards.append(ep["r"])
            self._ep_lengths.append(ep["l"])
            self._ep_count += 1

            mean_r = float(np.mean(self._ep_rewards[-50:]))
            mean_l = float(np.mean(self._ep_lengths[-50:]))

            row = {
                "experiment_id":     self.experiment_id,
                "timestep":          self.num_timesteps,
                "episode":           self._ep_count,
                "mean_reward_50":    round(mean_r, 3),
                "mean_ep_length_50": round(mean_l, 1),
                "timestamp":         datetime.now().isoformat(timespec="seconds"),
            }
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)

            if self.verbose >= 1:
                print(
                    f"  [Exp {self.experiment_id:>2}]  "
                    f"Ep {self._ep_count:>4}  |  "
                    f"Step {self.num_timesteps:>9,}  |  "
                    f"MeanRew(50): {mean_r:6.2f}  |  "
                    f"MeanLen(50): {mean_l:6.1f}"
                )
        return True


#  ENVIRONMENT
def make_env(n_envs: int = N_ENVS, seed: int = 42) -> VecFrameStack:
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)
    return env


#  TRAINING ONE EXPERIMENT
def train_experiment(
    exp: dict,
    policy: str = "CnnPolicy",
    total_timesteps: int = TOTAL_TIMESTEPS,
    verbose: int = 1,
) -> dict:
    exp_id       = exp["id"]
    label        = exp["label"]
    eps_fraction = exp["eps_d"] / total_timesteps

    print(f"  EXPERIMENT {exp_id:>2}: {label}")
    print(f"  lr={exp['lr']}  gamma={exp['gamma']}  batch={exp['batch']}")
    print(f"  epsilon: {exp['eps_s']} → {exp['eps_e']} over {exp['eps_d']:,} steps")
    print(f"  Policy: {policy}  |  Device: {device.upper()}  |  Steps: {total_timesteps:,}")

    train_env = make_env(seed=42)
    eval_env  = make_env(n_envs=1, seed=0)

    csv_path  = os.path.join(LOG_DIR, f"exp_{exp_id:02d}_rewards.csv")
    reward_cb = RewardLogger(csv_path, experiment_id=exp_id, verbose=verbose)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(LOG_DIR, f"exp_{exp_id:02d}_best"),
        log_path             = os.path.join(LOG_DIR, f"exp_{exp_id:02d}_eval"),
        eval_freq            = max(25_000 // N_ENVS, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 0,
    )

    model = DQN(
        policy                  = policy,
        env                     = train_env,
        device                  = device,
        learning_rate           = exp["lr"],
        gamma                   = exp["gamma"],
        batch_size              = exp["batch"],
        exploration_initial_eps = exp["eps_s"],
        exploration_final_eps   = exp["eps_e"],
        exploration_fraction    = eps_fraction,
        buffer_size             = 10_000,
        learning_starts         = 5_000,
        target_update_interval  = 1_000,
        train_freq              = 4,
        gradient_steps          = 1,
        optimize_memory_usage   = False,
        tensorboard_log         = TENSORBOARD_LOG,
        verbose                 = 0,
        seed                    = 42,
    )

    t0 = time.time()
    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [reward_cb, eval_cb],
        tb_log_name         = f"DQN_exp_{exp_id:02d}_{label.replace(' ', '_')}",
        reset_num_timesteps = True,
    )
    elapsed = time.time() - t0

    model_path = os.path.join(LOG_DIR, f"dqn_exp_{exp_id:02d}")
    model.save(model_path)
    print(f"  Saved → {model_path}.zip")

    train_env.close()
    eval_env.close()

    last_rewards = reward_cb._ep_rewards[-50:] or [0.0]
    mean_rew     = float(np.mean(last_rewards))

    result = {
        "experiment_id":      exp_id,
        "label":              label,
        "lr":                 exp["lr"],
        "gamma":              exp["gamma"],
        "batch_size":         exp["batch"],
        "eps_start":          exp["eps_s"],
        "eps_end":            exp["eps_e"],
        "eps_decay_steps":    exp["eps_d"],
        "policy":             policy,
        "mean_reward_last50": round(mean_rew, 3),
        "training_time_s":    round(elapsed, 1),
        "total_episodes":     reward_cb._ep_count,
        "noted_behavior":     NOTED_BEHAVIOR.get(exp_id, ""),
    }

    print(
        f"\n  ★ Done — Mean Reward (last 50): {mean_rew:.2f}  |  "
        f"Time: {elapsed:.0f}s  |  Episodes: {reward_cb._ep_count}"
    )
    return result


#  POLICY COMPARISON
def compare_policies(timesteps: int = 50_000) -> None:
    print("  POLICY COMPARISON: CnnPolicy vs MlpPolicy")

    scores = {}
    for pol in ("CnnPolicy", "MlpPolicy"):
        print(f"\n  Training {pol} for {timesteps:,} steps ...")
        exp = {**EXPERIMENTS[0], "id": 99, "label": f"Compare-{pol}"}
        r   = train_experiment(exp, policy=pol, total_timesteps=timesteps, verbose=0)
        scores[pol] = r["mean_reward_last50"]

    print("\n  ── Scores ──")
    for pol, sc in scores.items():
        print(f"  {pol:<12}: mean reward = {sc:.2f}")

    print("\n  ── Analysis ──")
    for pol, note in POLICY_NOTES.items():
        print(f"\n  {pol}:\n    {note}")

    winner = max(scores, key=scores.get)
    loser  = min(scores, key=scores.get)
    print(f"\n  Winner: {winner}  ({scores[winner]:.2f} vs {scores[loser]:.2f})")
    print("  Using CnnPolicy for all 10 experiments.\n")


#  PRINTING HELPERS
def print_table(results: list) -> None:
    hdr = f"{'#':<3} {'lr':<8} {'gamma':<7} {'batch':<6} {'e_s':<5} {'e_e':<5} {'e_decay':<9} {'reward':<8} label"
    sep = "─" * len(hdr)
    print("\n  HYPERPARAMETER TABLE — Person 1 (Carine)")
    print("  " + sep)
    print("  " + hdr)
    print("  " + sep)
    for r in results:
        print(
            f"  {r['experiment_id']:<3} "
            f"{str(r['lr']):<8} "
            f"{r['gamma']:<7} "
            f"{r['batch_size']:<6} "
            f"{r['eps_start']:<5} "
            f"{r['eps_end']:<5} "
            f"{r['eps_decay_steps']:<9} "
            f"{r['mean_reward_last50']:<8.2f} "
            f"{r['label']}"
        )
    print("  " + sep + "\n")


def print_behavior(results: list) -> None:
    print("  NOTED BEHAVIOR PER EXPERIMENT")
    for r in results:
        print(
            f"  Exp {r['experiment_id']:>2}  |  "
            f"Reward {r['mean_reward_last50']:>6.2f}  |  "
            f"{r['noted_behavior']}"
        )
    print()


# MAIN
def main(
    policy: str = "CnnPolicy",
    timesteps: int = TOTAL_TIMESTEPS,
    run_compare: bool = False,
    exp_id: int = None,
    verbose: int = 1,
) -> None:

    print(f"  DQN TRAINING — {ENV_ID}")
    print(f"  Policy  : {policy}   |   Steps/exp : {timesteps:,}")
    print(f"  Device  : {device.upper()}")
    print(f"  Saving  : {LOG_DIR}")

    if run_compare:
        compare_policies(timesteps=min(timesteps, 50_000))

    exps = (
        [e for e in EXPERIMENTS if e["id"] == exp_id]
        if exp_id else EXPERIMENTS
    )

    all_results = []
    for exp in exps:
        r = train_experiment(exp, policy=policy,
                             total_timesteps=timesteps, verbose=verbose)
        all_results.append(r)

    if not all_results:
        print("No experiments matched.")
        return

    # Save summary CSV
    summary_path = os.path.join(LOG_DIR, "experiments_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Summary CSV → {summary_path}")

    print_table(all_results)
    print_behavior(all_results)

    # Leaderboard
    ranked = sorted(all_results, key=lambda r: r["mean_reward_last50"], reverse=True)
    print("  LEADERBOARD")
    for rank, r in enumerate(ranked, 1):
        print(
            f"  #{rank:>2}  Exp {r['experiment_id']:>2}  "
            f"Reward: {r['mean_reward_last50']:>6.2f}  |  "
            f"lr={r['lr']}  gamma={r['gamma']}  batch={r['batch_size']}"
        )

    # Save best model as results/dqn_model.zip
    best     = ranked[0]
    best_src = os.path.join(LOG_DIR, f"dqn_exp_{best['experiment_id']:02d}.zip")
    best_model = DQN.load(best_src, device=device)
    best_model.save(MODEL_SAVE_PATH)

    print(f"\n  BEST: Experiment {best['experiment_id']} — {best['label']}")
    print(f"    lr={best['lr']}  gamma={best['gamma']}  batch={best['batch_size']}")
    print(f"    Mean Reward: {best['mean_reward_last50']:.2f}")
    print(f"\n  Saved → {MODEL_SAVE_PATH}.zip")
    print(f"  Syncs to your PC at: C:\\Users\\PC\\Desktop\\LTP\\deep-q-learning-formative-3\\results\\dqn_model.zip")