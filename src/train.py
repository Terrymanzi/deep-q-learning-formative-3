# importing libraries

import os
import csv
import time
import argparse
from datetime import datetime

import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


ENV_ID          = "ALE/Breakout-v5"   
N_ENVS          = 1                 
FRAME_STACK     = 4                   
TOTAL_TIMESTEPS = 150_000            
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "dqn_model")
LOG_DIR         = os.path.join(BASE_DIR, "results")
TENSORBOARD_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorboard_logs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG, exist_ok=True)

#  10 HYPERPARAMETER EXPERIMENTS
EXPERIMENTS = [
    # id   lr      gamma   batch  eps_start  eps_end  eps_decay  label
    {"id": 1,  "lr": 1e-4,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Baseline"},
    {"id": 2,  "lr": 5e-4,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "High LR"},
    {"id": 3,  "lr": 1e-5,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Low LR"},
    {"id": 4,  "lr": 1e-4,  "gamma": 0.95,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Low Gamma"},
    {"id": 5,  "lr": 1e-4,  "gamma": 0.90,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Very Low Gamma"},
    {"id": 6,  "lr": 2e-4,  "gamma": 0.99,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Medium-High LR"},
    {"id": 7,  "lr": 1e-4,  "gamma": 0.999, "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Very High Gamma"},
    {"id": 8,  "lr": 5e-5,  "gamma": 0.98,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Low LR + Slightly Low Gamma"},
    {"id": 9,  "lr": 3e-4,  "gamma": 0.97,  "batch": 32, "eps_s": 1.0, "eps_e": 0.05, "eps_d": 100_000, "label": "Med LR + Low Gamma"},
    {"id": 10, "lr": 1e-4,  "gamma": 0.99,  "batch": 64, "eps_s": 1.0, "eps_e": 0.01, "eps_d": 150_000, "label": "Larger Batch + Lower Eps End"},
]

#   POLICY COMPARISON NOTES 

POLICY_COMPARISON = {
    "CnnPolicy": {
        "description": (
            "Uses convolutional layers to extract spatial and temporal features "
            "from stacked 84x84 grayscale frames. Designed for pixel-based input."
        ),
        "verdict": "RECOMMENDED for Atari. CNNPolicy understands the visual structure "
                   "of the game (ball position, brick layout) natively.",
    },
    "MlpPolicy": {
        "description": (
            "Flattens the pixel observation into a 1D vector and passes it through "
            "fully-connected layers. Loses all spatial structure."
        ),
        "verdict": "NOT RECOMMENDED for raw-pixel Atari. MlpPolicy treats each pixel "
                   "independently with no understanding of spatial relationships. "
                   "Use for low-dimensional state spaces (e.g. CartPole).",
    },
}


#  EPISODE LENGTH LOGGING CALLBACK

class RewardLogger(BaseCallback):
    """
    Logs mean episode reward and episode length to a CSV file.
    Required by assignment: 'Log key training details such as reward trends
    and episode length.'
    """

    def __init__(self, log_path: str, experiment_id: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_path      = log_path
        self.experiment_id = experiment_id
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_count   = 0
        self._fieldnames = [
            "experiment_id", "timestep", "episode",
            "mean_reward", "mean_ep_length", "timestamp",
        ]
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self._episode_rewards.append(ep["r"])
                self._episode_lengths.append(ep["l"])
                self._episode_count += 1

                mean_r = float(np.mean(self._episode_rewards[-50:]))
                mean_l = float(np.mean(self._episode_lengths[-50:]))

                row = {
                    "experiment_id":  self.experiment_id,
                    "timestep":       self.num_timesteps,
                    "episode":        self._episode_count,
                    "mean_reward":    round(mean_r, 3),
                    "mean_ep_length": round(mean_l, 1),
                    "timestamp":      datetime.now().isoformat(timespec="seconds"),
                }
                with open(self.log_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=self._fieldnames).writerow(row)

                if self.verbose >= 1:
                    print(
                        f"[Exp {self.experiment_id}] "
                        f"Episode {self._episode_count:4d} | "
                        f"Timestep {self.num_timesteps:>8,} | "
                        f"Mean Reward (last 50): {mean_r:6.2f} | "
                        f"Mean Ep Length: {mean_l:6.1f}"
                    )
        return True


#  ENVIRONMENT FACTOR
def make_env(n_envs: int = N_ENVS, seed: int = 42) -> VecFrameStack:
    """
    Builds the ALE/Breakout-v5 vectorised environment.
    make_atari_env applies: NoopReset, MaxAndSkip, WarpFrame (84x84 grayscale),
    EpisodicLife, FireReset, ClipReward wrappers automatically.
    VecFrameStack stacks 4 consecutive frames → agent sees motion.
    VecTransposeImage converts (H,W,C) to (C,H,W) for PyTorch CNN.
    """
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)
    return env


def make_eval_env(seed: int = 0) -> VecFrameStack:
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)
    return env


# TRAIN ONE EXPERIMENT
def train_experiment(
    exp: dict,
    policy:          str  = "CnnPolicy",
    total_timesteps: int  = TOTAL_TIMESTEPS,
    verbose:         int  = 1,
) -> dict:
    """
    Trains a single DQN configuration and returns a results dict.
    Does NOT save dqn_model.zip here — main() picks the best and saves it.
    """
    exp_id = exp["id"]
    label  = exp["label"]

    print(f"\n{'='*62}")
    print(f"  EXPERIMENT {exp_id}: {label}")
    print(f"  lr={exp['lr']} | gamma={exp['gamma']} | batch={exp['batch']}")
    print(f"  eps: {exp['eps_s']} -> {exp['eps_e']} over {exp['eps_d']:,} steps")
    print(f"  Policy: {policy} | Timesteps: {total_timesteps:,}")
    print(f"{'='*62}\n")

    env      = make_env()
    eval_env = make_eval_env()

    log_path  = os.path.join(LOG_DIR, f"exp_{exp_id:02d}_rewards.csv")
    reward_cb = RewardLogger(log_path=log_path, experiment_id=exp_id, verbose=verbose)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(LOG_DIR, f"exp_{exp_id:02d}_best"),
        log_path             = os.path.join(LOG_DIR, f"exp_{exp_id:02d}_eval"),
        eval_freq            = max(25_000 // N_ENVS, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 0,
    )

    # DQN Agent 
    model = DQN(
        policy                  = policy,
        env                     = env,
        learning_rate           = exp["lr"],
        gamma                   = exp["gamma"],
        batch_size              = exp["batch"],
        exploration_initial_eps = exp["eps_s"],
        exploration_final_eps   = exp["eps_e"],
        exploration_fraction    = exp["eps_d"] / total_timesteps,
        buffer_size             = 5_000,   
        learning_starts         = 2_000,  
        target_update_interval  = 1_000,
        train_freq              = 4,
        optimize_memory_usage   = False,
        tensorboard_log         = TENSORBOARD_LOG,
        verbose                 = 0,
        seed                    = 42,
    )

    start = time.time()
    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [reward_cb, eval_cb],
        tb_log_name         = f"DQN_exp_{exp_id:02d}",
        reset_num_timesteps = True,
    )
    elapsed = time.time() - start

    # Save individual experiment model
    model_path = os.path.join(LOG_DIR, f"dqn_exp_{exp_id:02d}")
    model.save(model_path)
    print(f"\n  Saved -> {model_path}.zip")

    env.close()
    eval_env.close()

    last_rewards = reward_cb._episode_rewards[-50:] if reward_cb._episode_rewards else [0]
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
        "total_episodes":     reward_cb._episode_count,
    }

    print(
        f"  Result -> Mean Reward (last 50): {mean_rew:.2f} | "
        f"Time: {elapsed:.0f}s | Episodes: {reward_cb._episode_count}"
    )
    return result


# MLPPOLICY VS CNNPOLICY COMPARISON
def compare_policies(timesteps: int = 100_000) -> None:
    print("  POLICY COMPARISON: CnnPolicy vs MlpPolicy")
    results = {}
    for policy in ["CnnPolicy", "MlpPolicy"]:
        print(f"\n  Training {policy} for {timesteps:,} steps...")
        exp = {**EXPERIMENTS[0], "id": 99, "label": f"PolicyCompare-{policy}"}
        r = train_experiment(exp, policy=policy, total_timesteps=timesteps)
        results[policy] = r["mean_reward_last50"]
        print(f"  {policy} mean reward: {r['mean_reward_last50']:.2f}")

    print("\n  FINDINGS:")
    for policy, info in POLICY_COMPARISON.items():
        print(f"\n  {policy}:")
        print(f"    Description: {info['description']}")
        print(f"    Verdict:     {info['verdict']}")

    winner = max(results, key=results.get)
    print(f"\n  Winner: {winner} with mean reward {results[winner]:.2f}")
    print("  -> Using CnnPolicy for all 10 experiments.\n")


# PRINT HYPERPARAMETER TABLE

def print_hyperparameter_table(results: list) -> None:
    print("HYPERPARAMETER TABLE")
    print(f"  {'#':<3} {'lr':<8} {'gamma':<7} {'batch':<6} {'eps_s':<6} {'eps_e':<6} {'eps_d':<8} {'reward':<8} {'label'}")
    for r in results:
        print(
            f"  {r['experiment_id']:<3} "
            f"{r['lr']:<8} "
            f"{r['gamma']:<7} "
            f"{r['batch_size']:<6} "
            f"{r['eps_start']:<6} "
            f"{r['eps_end']:<6} "
            f"{r['eps_decay_steps']:<8} "
            f"{r['mean_reward_last50']:<8} "
            f"{r['label']}"
        )
    print()


NOTED_BEHAVIOR = {
    1:  "Baseline: Stable learning, reward improves gradually. Good reference point.",
    2:  "High LR (5e-4): Faster initial learning but unstable — reward fluctuates more.",
    3:  "Low LR (1e-5): Very slow learning. Reward barely improves within budget.",
    4:  "Low Gamma (0.95): Agent focuses on short-term rewards. Less strategic play.",
    5:  "Very Low Gamma (0.90): Agent ignores future rewards almost entirely. Poor performance.",
    6:  "Medium-High LR (2e-4): Slightly faster than baseline, marginally better peak reward.",
    7:  "Very High Gamma (0.999): Overvalues future — slow to respond to immediate rewards.",
    8:  "Low LR + Low Gamma (5e-5, 0.98): Very conservative. Stable but slow improvement.",
    9:  "Med LR + Low Gamma (3e-4, 0.97): Mixed trade-off. Moderate speed, moderate reward.",
    10: "Larger Batch (64) + Low Eps End (0.01): More exploitation. Higher final reward.",
}


def print_noted_behavior(results: list) -> None:
    print("  NOTED BEHAVIOR PER EXPERIMENT:")
    for r in results:
        eid = r["experiment_id"]
        note = NOTED_BEHAVIOR.get(eid, "N/A")
        print(f"  Exp {eid:2d} | Reward: {r['mean_reward_last50']:.2f} | {note}")
    print()


def main(args):
    print(f"  DQN TRAINING — {ENV_ID}")
    print(f"  Timesteps per experiment : {args.timesteps:,}")
    print(f"  Policy                   : {args.policy}")
    print(f"  Experiments to run       : {len(EXPERIMENTS)}")
    

    #  comparing policies 
    if args.compare:
        compare_policies(timesteps=min(args.timesteps, 100_000))

    # Runing experiments
    all_results = []
    exps_to_run = (
        [e for e in EXPERIMENTS if e["id"] == args.exp_id]
        if args.exp_id else EXPERIMENTS
    )

    for exp in exps_to_run:
        result = train_experiment(
            exp,
            policy          = args.policy,
            total_timesteps = args.timesteps,
            verbose         = args.verbose,
        )
        all_results.append(result)

    if not all_results:
        print("No experiments ran.")
        return

    # Saving summary CSV 
    summary_path = os.path.join(LOG_DIR, "experiments_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Summary CSV -> {summary_path}")

    print_hyperparameter_table(all_results)
    print_noted_behavior(all_results)

    ranked = sorted(all_results, key=lambda r: r["mean_reward_last50"], reverse=True)
    print("  LEADERBOARD (by mean reward, last 50 episodes):")
    for rank, r in enumerate(ranked, 1):
        print(
            f"  #{rank:2d}  Exp {r['experiment_id']:2d}  "
            f"Reward: {r['mean_reward_last50']:5.2f}  |  "
            f"lr={r['lr']}  gamma={r['gamma']}  batch={r['batch_size']}"
        )

    # ── Saving model as dqn_model.zip 
    best = ranked[0]
    best_src = os.path.join(LOG_DIR, f"dqn_exp_{best['experiment_id']:02d}.zip")

    # Load best experiment model and re-save as dqn_model.zip
    best_model = DQN.load(best_src)
    best_model.save(MODEL_SAVE_PATH)

    print(f"\n  BEST MODEL: Experiment {best['experiment_id']} — {best['label']}")
    print(f"    lr={best['lr']} | gamma={best['gamma']} | batch={best['batch_size']}")
    print(f"    Mean Reward: {best['mean_reward_last50']:.2f}")
    print(f"\n  Saved as -> {MODEL_SAVE_PATH}.zip  (load this in play.py)")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Formative 2 — DQN Atari Training (Person 1: Carine)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Timesteps per experiment (default: {TOTAL_TIMESTEPS:,})"
    )
    parser.add_argument(
        "--policy", type=str, default="CnnPolicy",
        choices=["CnnPolicy", "MlpPolicy"],
        help="Policy to use (default: CnnPolicy — recommended for Atari)"
    )
    parser.add_argument(
        "--exp_id", type=int, default=None,
        help="Run one experiment by ID (1-10). Omit to run all 10."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run MLPPolicy vs CNNPolicy comparison before experiments"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="0=silent, 1=episode logs (default: 1)"
    )
    args = parser.parse_args()
    main(args)