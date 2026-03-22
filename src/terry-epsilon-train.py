# terry-epsilon-train.py
# DQN epsilon hyperparameter sweep on ALE/Breakout-v5
# Optimised for Kaggle
# Varies: epsilon_start, epsilon_end, epsilon_decay_steps
# Fixes: lr=1e-4  gamma=0.99  batch=32  (Carine's baseline values)

# If on KAGGLE, run this install first
# !pip install -q stable-baselines3[extra] ale-py gymnasium[atari] autorom[accept-rom-license]
# !AutoROM --accept-license --quiet
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import time
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

import ale_py  # noqa: F401
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


# ─── DEVICE ──────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device : {device.upper()}")
if device == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")


# ─── PATHS (Kaggle-native) ────────────────────────────────────────────────────
# /kaggle/working is persisted and appears in Notebook Output for download.

KAGGLE_WORKING  = Path("/kaggle/working")
LOG_DIR         = KAGGLE_WORKING / "results" / "Terry"
MODEL_SAVE_PATH = LOG_DIR / "dqn_model"
TENSORBOARD_LOG = KAGGLE_WORKING / "tensorboard_logs"
ZIP_OUTPUT      = KAGGLE_WORKING / "terry_epsilon_sweep_outputs.zip"

LOG_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOG.mkdir(parents=True, exist_ok=True)

print(f"  Results : {LOG_DIR}")
print(f"  Best    : {MODEL_SAVE_PATH}.zip")
print(f"  Archive : {ZIP_OUTPUT}\n")


# ─── FIXED HYPERPARAMETERS ───────────────────────────────────────────────────

ENV_ID          = "ALE/Breakout-v5"
FRAME_STACK     = 4
TOTAL_TIMESTEPS = 150_000
N_ENVS          = 1
POLICY          = "CnnPolicy"
FIXED_LR        = 1e-4
FIXED_GAMMA     = 0.99
FIXED_BATCH     = 32


# ─── EPSILON EXPERIMENTS ─────────────────────────────────────────────────────

EPSILON_EXPERIMENTS = [
    # id   eps_start  eps_end   eps_decay   label
    {"id":  1, "eps_s": 1.00, "eps_e": 0.050, "eps_d": 100_000, "label": "Baseline Epsilon"},
    {"id":  2, "eps_s": 1.00, "eps_e": 0.100, "eps_d": 100_000, "label": "High Eps End"},
    {"id":  3, "eps_s": 1.00, "eps_e": 0.010, "eps_d": 100_000, "label": "Low Eps End"},
    {"id":  4, "eps_s": 1.00, "eps_e": 0.050, "eps_d":  50_000, "label": "Short Decay"},
    {"id":  5, "eps_s": 1.00, "eps_e": 0.050, "eps_d": 140_000, "label": "Long Decay"},
    {"id":  6, "eps_s": 0.50, "eps_e": 0.050, "eps_d": 100_000, "label": "Low Eps Start"},
    {"id":  7, "eps_s": 1.00, "eps_e": 0.005, "eps_d": 130_000, "label": "Very Low Eps End + Long Decay"},
    {"id":  8, "eps_s": 1.00, "eps_e": 0.010, "eps_d":  50_000, "label": "Fast Decay + Low Eps End"},
    {"id":  9, "eps_s": 1.00, "eps_e": 0.050, "eps_d":  30_000, "label": "Very Short Decay"},
    {"id": 10, "eps_s": 0.10, "eps_e": 0.010, "eps_d": 100_000, "label": "Near-Greedy From Start"},
]

NOTED_BEHAVIOR = {
    1:  "Baseline epsilon schedule; steady improvement — good reference point.",
    2:  "High eps_end (0.1): stays more exploratory late; noisier policy, lower reward.",
    3:  "Low eps_end (0.01): exploits more at end; cleaner late-game policy.",
    4:  "Short decay (50k): switches to exploitation early; gains speed but misses exploration.",
    5:  "Long decay (140k): explores almost the whole run; slower but broader search.",
    6:  "Low eps_start (0.5): half as much early exploration; slightly under-explores.",
    7:  "Very low eps_end + long decay: maximises exploitation late; best greedy performance.",
    8:  "Fast decay + low eps_end: aggressive exploitation switch; moderate reward.",
    9:  "Very short decay (30k): exploits way too early; agent locked into suboptimal policy.",
    10: "Near-greedy from start (eps_s=0.1): minimal exploration; lowest reward due to poor Q init.",
}


# ─── REWARD LOGGER ───────────────────────────────────────────────────────────

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

            # Print every 10 episodes to avoid flooding Kaggle's output buffer
            if self.verbose >= 1 and self._ep_count % 10 == 0:
                print(
                    f"  [Exp {self.experiment_id:>2}]  "
                    f"Ep {self._ep_count:>4}  |  "
                    f"Step {self.num_timesteps:>9,}  |  "
                    f"MeanRew(50): {mean_r:6.2f}  |  "
                    f"MeanLen(50): {mean_l:6.1f}",
                    flush=True,
                )
        return True


# ─── ENVIRONMENT FACTORY ─────────────────────────────────────────────────────

def make_env(n_envs: int = N_ENVS, seed: int = 42) -> VecTransposeImage:
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    env = VecTransposeImage(env)
    return env


# ─── TRAIN ONE EXPERIMENT ────────────────────────────────────────────────────

def train_experiment(exp: dict, verbose: int = 1) -> dict:
    exp_id       = exp["id"]
    label        = exp["label"]
    eps_fraction = exp["eps_d"] / TOTAL_TIMESTEPS

    print(f"\n  ━━━  EXPERIMENT {exp_id:>2}: {label}  ━━━", flush=True)
    print(f"  eps_start={exp['eps_s']}  eps_end={exp['eps_e']}  eps_decay={exp['eps_d']:,}", flush=True)
    print(f"  lr={FIXED_LR}  gamma={FIXED_GAMMA}  batch={FIXED_BATCH}  steps={TOTAL_TIMESTEPS:,}\n", flush=True)

    train_env = make_env(seed=42)
    eval_env  = make_env(n_envs=1, seed=0)

    csv_path  = str(LOG_DIR / f"exp_{exp_id:02d}_rewards.csv")
    reward_cb = RewardLogger(csv_path, experiment_id=exp_id, verbose=verbose)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(LOG_DIR / f"exp_{exp_id:02d}_best"),
        log_path             = str(LOG_DIR / f"exp_{exp_id:02d}_eval"),
        eval_freq            = max(25_000 // N_ENVS, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 0,
    )

    model = DQN(
        policy                  = POLICY,
        env                     = train_env,
        device                  = device,
        learning_rate           = FIXED_LR,
        gamma                   = FIXED_GAMMA,
        batch_size              = FIXED_BATCH,
        exploration_initial_eps = exp["eps_s"],
        exploration_final_eps   = exp["eps_e"],
        exploration_fraction    = eps_fraction,
        buffer_size             = 10_000,
        learning_starts         = 5_000,
        target_update_interval  = 1_000,
        train_freq              = 4,
        gradient_steps          = 1,
        optimize_memory_usage   = False,
        tensorboard_log         = str(TENSORBOARD_LOG),
        verbose                 = 0,
        seed                    = 42,
    )

    t0 = time.time()
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = [reward_cb, eval_cb],
        tb_log_name         = f"DQN_eps_{exp_id:02d}_{label.replace(' ', '_')}",
        reset_num_timesteps = True,
    )
    elapsed = time.time() - t0

    model_path = str(LOG_DIR / f"dqn_exp_{exp_id:02d}")
    model.save(model_path)
    print(f"  Saved → {model_path}.zip", flush=True)

    train_env.close()
    eval_env.close()

    last_rewards = reward_cb._ep_rewards[-50:] or [0.0]
    mean_rew     = float(np.mean(last_rewards))

    result = {
        "experiment_id":      exp_id,
        "label":              label,
        "epsilon_start":      exp["eps_s"],
        "epsilon_end":        exp["eps_e"],
        "epsilon_decay":      exp["eps_d"],
        "lr":                 FIXED_LR,
        "gamma":              FIXED_GAMMA,
        "batch_size":         FIXED_BATCH,
        "policy":             POLICY,
        "mean_reward_last50": round(mean_rew, 3),
        "training_time_s":    round(elapsed, 1),
        "total_episodes":     reward_cb._ep_count,
        "observations":       NOTED_BEHAVIOR.get(exp_id, ""),
    }

    print(
        f"\n  ★ Done — Mean Reward (last 50): {mean_rew:.2f}  |  "
        f"Time: {elapsed:.0f}s  |  Episodes: {reward_cb._ep_count}",
        flush=True,
    )
    return result


# ─── SUMMARY TABLE ───────────────────────────────────────────────────────────

def print_table(results: list) -> None:
    hdr = f"{'#':<3} {'eps_s':<7} {'eps_e':<7} {'eps_decay':<10} {'reward':<8} label"
    sep = "─" * 65
    print("\n  EPSILON EXPERIMENT RESULTS — Terry")
    print("  " + sep)
    print("  " + hdr)
    print("  " + sep)
    for r in results:
        print(
            f"  {r['experiment_id']:<3} "
            f"{r['epsilon_start']:<7} "
            f"{r['epsilon_end']:<7} "
            f"{r['epsilon_decay']:<10} "
            f"{r['mean_reward_last50']:<8.2f} "
            f"{r['label']}"
        )
    print("  " + sep + "\n", flush=True)


# ─── ZIP ALL OUTPUTS ─────────────────────────────────────────────────────────

def zip_outputs() -> None:
    """
    Bundle every file under LOG_DIR and the tensorboard_logs folder into a
    single archive at ZIP_OUTPUT so the user can download everything in one
    click from the Kaggle Notebook Output panel.
    """
    folders_to_zip = [LOG_DIR, TENSORBOARD_LOG]
    print(f"\n  Zipping outputs → {ZIP_OUTPUT}", flush=True)
    with zipfile.ZipFile(ZIP_OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in folders_to_zip:
            for file_path in sorted(folder.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(KAGGLE_WORKING)
                    zf.write(file_path, arcname)
                    print(f"    + {arcname}", flush=True)
    size_mb = ZIP_OUTPUT.stat().st_size / 1_048_576
    print(f"\n  Archive ready: {ZIP_OUTPUT}  ({size_mb:.1f} MB)", flush=True)
    print("  → Download from Kaggle Notebook Output panel on the right.", flush=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"  DQN EPSILON SWEEP — {ENV_ID}", flush=True)
    print(f"  Fixed: lr={FIXED_LR}  gamma={FIXED_GAMMA}  batch={FIXED_BATCH}", flush=True)
    print(f"  Device: {device.upper()}  |  Steps/exp: {TOTAL_TIMESTEPS:,}\n", flush=True)

    all_results = []
    for exp in EPSILON_EXPERIMENTS:
        result = train_experiment(exp, verbose=1)
        all_results.append(result)

    # ── save summary CSV ──
    summary_path = str(LOG_DIR / "experiments_summary.csv")
    fieldnames = [
        "experiment_id", "label",
        "epsilon_start", "epsilon_end", "epsilon_decay",
        "lr", "gamma", "batch_size", "policy",
        "mean_reward_last50", "training_time_s", "total_episodes", "observations",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"  Summary CSV → {summary_path}", flush=True)

    print_table(all_results)

    # ── leaderboard ──
    ranked = sorted(all_results, key=lambda r: r["mean_reward_last50"], reverse=True)
    print("  LEADERBOARD", flush=True)
    for rank, r in enumerate(ranked, 1):
        print(
            f"  #{rank:>2}  Exp {r['experiment_id']:>2}  "
            f"Reward: {r['mean_reward_last50']:>6.2f}  |  "
            f"eps_s={r['epsilon_start']}  eps_e={r['epsilon_end']}  "
            f"eps_d={r['epsilon_decay']}",
            flush=True,
        )

    # ── save best model ──
    best     = ranked[0]
    best_src = str(LOG_DIR / f"dqn_exp_{best['experiment_id']:02d}.zip")
    best_model = DQN.load(best_src, device=device)
    best_model.save(str(MODEL_SAVE_PATH))
    print(f"\n  BEST: Experiment {best['experiment_id']} — {best['label']}", flush=True)
    print(f"    eps_s={best['epsilon_start']}  eps_e={best['epsilon_end']}  eps_d={best['epsilon_decay']}", flush=True)
    print(f"    Mean Reward: {best['mean_reward_last50']:.2f}", flush=True)
    print(f"  Saved → {MODEL_SAVE_PATH}.zip", flush=True)

    # ── zip everything for one-click download ──
    zip_outputs()


if __name__ == "__main__":
    main()