import matplotlib

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Prefer existing reader from the orchestrator where available
try:
    from qlearning.orchestrator import read_metrics_since  # [`read_metrics_since`](qlearning/orchestrator.py)
except ModuleNotFoundError:
    from orchestrator import read_metrics_since
matplotlib.use("Agg")


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    """
    Load all JSONL entries from path (wrapper around read_metrics_since).
    """
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    # read_metrics_since expects an offset; pass 0 to read entire file
    return read_metrics_since(path, 0)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kern = np.ones(window, dtype=float) / window
    return np.convolve(x, kern, mode="valid")


def plot_metrics(
    metrics_path: Path,
    out_dir: Path,
    prefix: str,
    rolling_window: int = 100,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_metrics(metrics_path)
    if not entries:
        raise RuntimeError(f"No entries found in {metrics_path}")

    # Extract series (use .get to be robust to missing keys)
    game_ids = np.array([int(e.get("game_id", i)) for i, e in enumerate(entries)])
    alpha = np.array([float(e.get("alpha", np.nan)) for e in entries])
    epsilon = np.array([float(e.get("epsilon", np.nan)) for e in entries])
    td_error = np.array([float(e.get("avg_abs_td_error", np.nan)) for e in entries])
    delta_w = np.array([float(e.get("avg_abs_delta_w", np.nan)) for e in entries])
    shaped = np.array([float(e.get("sum_shaped_reward", np.nan)) for e in entries])
    terminal = np.array([float(e.get("sum_terminal_reward", np.nan)) for e in entries])
    weight_norm = np.array([float(e.get("weight_l1_norm", np.nan)) for e in entries])
    winners = [e.get("winner") for e in entries]

    # Determine defender win flag. The logged "role" in metrics is the player's role;
    # a winner equal to "defender" indicates defender won. Adjust if different convention.
    win_flags = np.array([1.0 if w == "defender" else 0.0 for w in winners], dtype=float)

    plots: List[Path] = []

    # Combined figure with multiple subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

    # Alpha & Epsilon
    axs[0].plot(game_ids, alpha, label="alpha", color="C0")
    axs[0].plot(game_ids, epsilon, label="epsilon", color="C1")
    axs[0].set_ylabel("learning rate / eps")
    axs[0].legend()
    axs[0].set_title("Alpha and Epsilon over Games")

    # TD error and delta_w (log scale if wide)
    axs[1].plot(game_ids, td_error, label="avg_abs_td_error", color="C2", alpha=0.9)
    axs[1].plot(game_ids, delta_w, label="avg_abs_delta_w", color="C3", alpha=0.9)
    axs[1].set_ylabel("magnitude")
    axs[1].legend()
    axs[1].set_title("TD Error and Weight Updates")

    # Rewards
    axs[2].plot(game_ids, shaped, label="sum_shaped_reward", color="C4", alpha=0.9)
    axs[2].plot(game_ids, terminal, label="sum_terminal_reward", color="C5", alpha=0.9)
    axs[2].set_ylabel("reward")
    axs[2].legend()
    axs[2].set_title("Shaped and Terminal Rewards per Game")

    # Weight norm and rolling win rate
    axs[3].plot(game_ids, weight_norm, label="weight_l1_norm", color="C6", alpha=0.9)
    if len(win_flags) >= rolling_window:
        wins_rm = rolling_mean(win_flags, rolling_window)
        games_rm = game_ids[rolling_window - 1 :]
        ax2 = axs[3].twinx()
        ax2.plot(games_rm, wins_rm, label=f"{rolling_window}-game win rate", color="k", alpha=0.8)
        ax2.set_ylabel("win rate")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")
    axs[3].set_ylabel("weight L1 norm")
    axs[3].legend(loc="upper left")
    axs[3].set_title("Weight Norm and Rolling Win Rate")

    combined_path = out_dir / f"{prefix}_metrics_combined.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    plots.append(combined_path)

    # Additional focused plots
    def save_simple(x, y, xlabel, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        p = out_dir / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p

    plots.append(save_simple(game_ids, td_error, "game", "avg_abs_td_error", "TD Error per Game", f"{prefix}_td_error.png"))
    plots.append(save_simple(game_ids, delta_w, "game", "avg_abs_delta_w", "Avg Abs Delta w per Game", f"{prefix}_delta_w.png"))
    plots.append(save_simple(game_ids, weight_norm, "game", "weight_l1_norm", "Weight L1 Norm per Game", f"{prefix}_weight_norm.png"))
    plots.append(save_simple(game_ids, win_flags, "game", "win (1=defender)", "Defender Win Flag per Game", f"{prefix}_win_flags.png"))

    return plots


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot qlearning defender training metrics")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(__file__).parent.parent / "qlearning" / "learning_metrics_defender.jsonl",
        help="path to defender learning metrics jsonl (default: qlearning/learning_metrics_defender.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "metrics/images",
        help="output directory to write plots (default: qlearning/metrics/images)",
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='prefix for the output files'
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="window size for rolling win rate",
    )
    args = parser.parse_args()

    # Normalize default path if user invoked from repo root
    log_path = args.log_file
    if not log_path.exists():
        # also try local qlearning default
        candidate = Path(__file__).parent / "learning_metrics_defender.jsonl"
        if candidate.exists():
            log_path = candidate

    plots = plot_metrics(log_path, args.out_dir, rolling_window=args.rolling_window, prefix=args.prefix)
    for p in plots:
        print(p)

